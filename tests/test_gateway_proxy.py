from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from fastapi.testclient import TestClient

import nmesh.gateway as gateway_module
from nmesh import telemetry
from nmesh.bench import benchmark_key, measure
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app, route
from nmesh.planner import Plan, Policy, build_plan, save_plan

from .test_planner import profile


class _UpstreamHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        self.wfile.write(b'data: {"choices":[{"delta":{}}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: object) -> None:
        return


def test_gateway_proxy_rewrites_model_and_forwards_sse() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("proxy-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        client = TestClient(create_app(plan))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert _UpstreamHandler.request_body["model"] == service.model_ref
        assert "data: [DONE]" in response.text
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_uses_streaming_endpoint() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("bench-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        result = measure(plan.services[0], f"http://127.0.0.1:{upstream.server_address[1]}",
                         prefill_tokens=8, decode_tokens=2, runs=3)
        assert result.prefill_tps > 0
        assert result.decode_tps > 0
        assert _UpstreamHandler.request_body["max_tokens"] == 2
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_explicit_service_model_wins_over_code_heuristic() -> None:
    model = ModelSpec("chat-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    assert route({
        "model": "nmesh-chat",
        "messages": [{"role": "user", "content": "```python\nprint(1)\n```"}],
    }, plan) == "chat"
    assert route({
        "model": f"nmesh-{plan.services[0].name}",
        "messages": [{"role": "user", "content": "```python\nprint(1)\n```"}],
    }, plan) == plan.services[0].name


def test_auto_and_unknown_model_use_heuristics() -> None:
    chat_model = ModelSpec("chat-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                           4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    code_model = ModelSpec("code-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                           4096, ["code"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [chat_model, code_model],
                      Policy(roles=["chat", "code"]))
    request = {"messages": [{"role": "user", "content": "```python\nprint(1)\n```"}]}
    assert route({**request, "model": "nmesh-auto"}, plan) == plan.routing.role_to_service["code"]
    assert route({**request, "model": "nmesh-unknown"}, plan) == plan.routing.role_to_service["code"]
    assert route({"model": "nmesh-auto", "tools": [{"type": "function"}]}, plan) == (
        plan.routing.role_to_service["chat"]
    )


class _ReloadHandler(BaseHTTPRequestHandler):
    bodies: ClassVar[list[dict[str, object]]] = []
    block: ClassVar[bool] = False
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.bodies.append(body)
        if self.__class__.block:
            self.__class__.started.set()
            self.__class__.release.wait(timeout=5)
        payload = json.dumps({
            "model": body["model"],
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 3},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _reload_plans(port: int) -> tuple[Plan, Plan]:
    model = ModelSpec("reload-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    base = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = replace(base.services[0], port=port, model_ref="old-ref")
    old = replace(base, services=[service], created_at="old")
    new_service = replace(service, model_id="new-model", model_ref="new-ref", quant="q8_0")
    new = replace(old, services=[new_service], created_at="new")
    return old, new


def _start_reload_upstream() -> ThreadingHTTPServer:
    _ReloadHandler.bodies = []
    _ReloadHandler.block = False
    _ReloadHandler.started.clear()
    _ReloadHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ReloadHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    return upstream


def test_gateway_admin_reload_refreshes_plan_and_telemetry(
    tmp_path, monkeypatch
) -> None:
    upstream = _start_reload_upstream()
    try:
        old, new = _reload_plans(upstream.server_address[1])
        path = tmp_path / "plan.json"
        save_plan(old, path)
        monkeypatch.setattr(gateway_module, "PLAN_PATH", path)
        client = TestClient(create_app(old))

        assert client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}],
        }).json()["model"] == "nmesh-auto"
        old_gpu = old.profile.gpus[0].name
        old_key = benchmark_key(
            old.services[0].model_id, old.services[0].quant, old.services[0].backend,
            old_gpu, old.services[0].n_gpu_layers,
        )
        save_plan(new, path)
        response = client.post("/admin/reload")
        assert response.status_code == 200
        assert response.json() == {
            "reloaded": True, "services": ["chat"], "created_at": "new",
        }
        assert client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}],
        }).status_code == 200
        assert _ReloadHandler.bodies[0]["model"] == "old-ref"
        assert _ReloadHandler.bodies[1]["model"] == "new-ref"
        new_key = benchmark_key(
            new.services[0].model_id, new.services[0].quant, new.services[0].backend,
            old_gpu, new.services[0].n_gpu_layers,
        )
        samples = telemetry._default.samples()
        assert {sample.key for sample in samples} == {old_key, new_key}
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_gateway_auto_reload_uses_plan_mtime(tmp_path, monkeypatch) -> None:
    upstream = _start_reload_upstream()
    try:
        old, new = _reload_plans(upstream.server_address[1])
        path = tmp_path / "plan.json"
        save_plan(old, path)
        monkeypatch.setattr(gateway_module, "PLAN_PATH", path)
        client = TestClient(create_app())
        save_plan(new, path)
        mtime_ns = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(mtime_ns, mtime_ns))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert _ReloadHandler.bodies[-1]["model"] == "new-ref"
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_gateway_request_keeps_plan_snapshot_during_reload(tmp_path, monkeypatch) -> None:
    upstream = _start_reload_upstream()
    _ReloadHandler.block = True
    try:
        old, new = _reload_plans(upstream.server_address[1])
        path = tmp_path / "plan.json"
        save_plan(old, path)
        monkeypatch.setattr(gateway_module, "PLAN_PATH", path)
        client = TestClient(create_app(old))
        responses: list[object] = []
        thread = threading.Thread(target=lambda: responses.append(client.post(
            "/v1/chat/completions",
            json={"model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}]},
        )))
        thread.start()
        assert _ReloadHandler.started.wait(timeout=2)
        save_plan(new, path)
        reload_response = client.post("/admin/reload")
        assert reload_response.status_code == 200
        _ReloadHandler.release.set()
        thread.join(timeout=5)
        assert len(responses) == 1
        assert _ReloadHandler.bodies[0]["model"] == "old-ref"
    finally:
        _ReloadHandler.release.set()
        upstream.shutdown()
        upstream.server_close()


def test_gateway_watchdog_runs_and_cancels(monkeypatch) -> None:
    model = ModelSpec("watchdog-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    called = threading.Event()

    def fake_heartbeat() -> None:
        called.set()

    monkeypatch.setattr(gateway_module, "heartbeat", fake_heartbeat)
    with TestClient(create_app(plan, watchdog=True, watchdog_interval=0.01)):
        assert called.wait(timeout=2)


def test_gateway_retries_once_after_connect_error(monkeypatch) -> None:
    model = ModelSpec("retry-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = plan.services[0]
    calls: list[tuple[str, Plan]] = []

    def fake_ensure(name: str, snapshot: Plan) -> None:
        calls.append((name, snapshot))

    monkeypatch.setattr(gateway_module, "ensure_running", fake_ensure)
    client = TestClient(create_app(plan))
    response = client.post("/v1/chat/completions", json={
        "model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 502
    assert len(calls) == 1
    assert calls[0] == (service.name, plan)
