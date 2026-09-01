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


class _CompatStreamHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in (
            b": keep-alive\n\n",
            b'data: {"model":"upstream-model","choices":[{"delta":{"content":"ok"}}]}\r\n',
            b"\r\n",
            b"data: {not-json}\n\n",
            b"data: [DONE]\n\n",
        ):
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


class _UsageHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}
    usage_on_every_chunk: ClassVar[bool] = False

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index in range(20):
            data: dict[str, object] = {
                "choices": [{"delta": {"content": str(index)}}],
            }
            if self.__class__.usage_on_every_chunk:
                data["usage"] = {
                    "prompt_tokens": 23,
                    "completion_tokens": index + 1,
                }
            payload = json.dumps(data).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()
        payload = json.dumps({
            "choices": [],
            "usage": {"prompt_tokens": 23, "completion_tokens": 17},
        }).encode()
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: object) -> None:
        return


class _RawErrorHandler(BaseHTTPRequestHandler):
    body = b'{"error":{"code":400,"message":"upstream failure","type":"invalid_request_error"}}'

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

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


def test_gateway_stream_rewrites_model_and_preserves_sse_frames() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CompatStreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("compat-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        client = TestClient(create_app(replace(plan, services=[service])))
        response = client.post("/v1/chat/completions", json={
            "model": "client-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert b": keep-alive\n\n" in response.content
        assert b"\r\n\r\n" in response.content
        assert b"data: {not-json}\n\n" in response.content
        assert b"data: [DONE]\n\n" in response.content
        payloads = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
            and line != "data: {not-json}"
        ]
        assert payloads[0]["model"] == "client-model"
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_llamacpp_stream_injects_usage_filters_unrequested_chunk_and_records_exact(
    monkeypatch,
) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    samples = []
    monkeypatch.setattr(gateway_module, "record_telemetry", samples.append)
    try:
        model = ModelSpec("usage-proxy-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        client = TestClient(create_app(replace(plan, services=[service])))

        without_usage = client.post("/v1/chat/completions", json={
            "model": "client-model", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert without_usage.status_code == 200
        assert '"choices":[]' not in without_usage.text
        assert _UsageHandler.request_body["stream_options"] == {
            "include_usage": True,
        }
        assert samples[-1].approximate is False
        assert samples[-1].completion_tokens == 17

        with_usage = client.post("/v1/chat/completions", json={
            "model": "client-model", "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert with_usage.status_code == 200
        assert any(
            json.loads(line[6:]).get("choices") == []
            for line in with_usage.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        assert samples[-1].approximate is False
        assert samples[-1].completion_tokens == 17
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_non_llamacpp_stream_does_not_inject_usage(monkeypatch) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    original_base_url = gateway_module._base_url
    monkeypatch.setattr(
        gateway_module,
        "_base_url",
        lambda service: f"http://127.0.0.1:{upstream.server_address[1]}",
    )
    try:
        model = ModelSpec("backend-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        for backend in ("ollama", "vllm"):
            service = replace(plan.services[0], backend=backend)
            client = TestClient(create_app(replace(plan, services=[service])))
            response = client.post("/v1/chat/completions", json={
                "model": "client-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert response.status_code == 200
            assert "stream_options" not in _UpstreamHandler.request_body
    finally:
        monkeypatch.setattr(gateway_module, "_base_url", original_base_url)
        upstream.shutdown()
        upstream.server_close()


def test_upstream_openai_error_is_passed_through_unchanged() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _RawErrorHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("error-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        client = TestClient(create_app(replace(plan, services=[service])))
        response = client.post("/v1/chat/completions", json={
            "model": "client-model",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 400
        assert response.content == _RawErrorHandler.body
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
        assert result.approximate is True
        assert _UpstreamHandler.request_body["max_tokens"] == 2
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_uses_upstream_usage_counts() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _UsageHandler.usage_on_every_chunk = True
        model = ModelSpec("usage-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        result = measure(
            plan.services[0], f"http://127.0.0.1:{upstream.server_address[1]}",
            prefill_tokens=512, decode_tokens=17, runs=1,
        )
        assert result.approximate is False
        assert result.prompt_tokens == 23
        assert _UsageHandler.request_body["stream_options"] == {
            "include_usage": True,
        }
    finally:
        _UsageHandler.usage_on_every_chunk = False
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
    service = replace(
        plan.services[0],
        memory=replace(plan.services[0].memory, parallel_slots=1),
    )
    plan = replace(plan, services=[service])
    calls: list[tuple[str, Plan]] = []

    def fake_ensure(name: str, snapshot: Plan) -> None:
        calls.append((name, snapshot))

    monkeypatch.setattr(gateway_module, "ensure_running", fake_ensure)
    client = TestClient(create_app(plan))
    response = client.post("/v1/chat/completions", json={
        "model": "nmesh-auto", "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == 502
    assert len(calls) == 1
    assert calls[0] == (service.name, plan)
    assert client.get("/metrics").json()["concurrency"][service.name]["in_flight"] == 0
