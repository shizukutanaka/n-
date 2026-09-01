from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

from fastapi.testclient import TestClient

import nmesh.gateway as gateway_module
from nmesh import cli, telemetry
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app, route
from nmesh.planner import Policy, build_plan

from .test_planner import profile


class _CompletionHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict[str, object]]]] = []
    status: ClassVar[int] = 200
    block: ClassVar[bool] = False
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append((self.path, body))
        type(self).started.set()
        if type(self).block:
            type(self).release.wait(timeout=5)
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"text":"ok"}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        payload = json.dumps({
            "model": "upstream-model",
            "choices": [{"text": "ok"}],
            "usage": {"completion_tokens": 16},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _completion_upstream() -> ThreadingHTTPServer:
    _CompletionHandler.requests = []
    _CompletionHandler.status = 200
    _CompletionHandler.block = False
    _CompletionHandler.started.clear()
    _CompletionHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    return upstream


def _completion_plan(port: int, slots: int = 2):
    model = ModelSpec(
        "completion-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        port=port,
        memory=replace(plan.services[0].memory, parallel_slots=slots),
    )
    return replace(plan, services=[service])


def test_legacy_completions_proxy_and_streaming() -> None:
    upstream = _completion_upstream()
    try:
        plan = _completion_plan(upstream.server_address[1])
        with TestClient(create_app(plan)) as client:
            response = client.post(
                "/v1/completions",
                json={"model": f"nmesh-{plan.services[0].name}", "prompt": "hello"},
            )
            assert response.status_code == 200
            assert response.json()["model"] == f"nmesh-{plan.services[0].name}"
            assert _CompletionHandler.requests[-1][0] == "/v1/completions"
            assert _CompletionHandler.requests[-1][1]["model"] == plan.services[0].model_ref

            streamed = client.post(
                "/v1/completions",
                json={"model": "nmesh-auto", "prompt": ["hello", "world"], "stream": True},
            )
            assert streamed.status_code == 200
            assert "data: [DONE]" in streamed.text
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_completion_prompt_routing_uses_string_and_list_content() -> None:
    model = ModelSpec(
        "routing-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    small = replace(plan.services[0], name="small", context=100)
    large = replace(plan.services[0], name="large", context=1000)
    plan = replace(
        plan,
        services=[small, large],
        routing=replace(plan.routing, role_to_service={"chat": "small"}),
    )
    assert route({"prompt": "x" * 400}, plan) == "large"
    assert route({"prompt": ["x" * 200, "y" * 200]}, plan) == "large"


def test_completion_slot_limiter_matches_chat(monkeypatch) -> None:
    upstream = _completion_upstream()
    try:
        plan = _completion_plan(upstream.server_address[1], slots=1)
        monkeypatch.setattr(gateway_module, "QUEUE_TIMEOUT", 0.01)
        _CompletionHandler.block = True
        with TestClient(create_app(plan)) as client:
            result: list[object] = []
            first = threading.Thread(target=lambda: result.append(
                client.post("/v1/completions", json={"prompt": "first"})
            ))
            first.start()
            assert _CompletionHandler.started.wait(timeout=2)
            second = client.post("/v1/completions", json={"prompt": "second"})
            assert second.status_code == 503
            assert second.headers["Retry-After"] == "1"
            _CompletionHandler.release.set()
            first.join(timeout=5)
            assert result[0].status_code == 200
    finally:
        _CompletionHandler.release.set()
        upstream.shutdown()
        upstream.server_close()


def test_completion_404_names_backend() -> None:
    upstream = _completion_upstream()
    try:
        plan = _completion_plan(upstream.server_address[1])
        _CompletionHandler.status = 404
        with TestClient(create_app(plan)) as client:
            response = client.post("/v1/completions", json={"prompt": "hello"})
        assert response.status_code == 502
        assert "llamacpp" in response.json()["detail"]
    finally:
        _CompletionHandler.status = 200
        upstream.shutdown()
        upstream.server_close()


def test_api_key_authentication(monkeypatch) -> None:
    monkeypatch.delenv("NMESH_API_KEY", raising=False)
    plan = _completion_plan(1)
    with TestClient(create_app(plan)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    secret = "test-secret"
    monkeypatch.setenv("NMESH_API_KEY", secret)
    with TestClient(create_app(plan)) as client:
        assert client.get("/health").status_code == 200
        missing = client.get("/metrics")
        wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        right = client.get("/v1/models", headers={"Authorization": f"Bearer {secret}"})
        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert right.status_code == 200
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        assert secret not in missing.text
        assert secret not in wrong.text


def test_prometheus_metrics_have_help_type_labels_and_escaping() -> None:
    telemetry.record(telemetry.Sample(
        'quoted"service\\line\nbreak', "key", 12.0, 0.1, 0.2, 16, 1.0, True,
    ))
    plan = _completion_plan(1)
    with TestClient(create_app(plan)) as client:
        response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'service="quoted\\"service\\\\line\\nbreak"' in text
    assert 'approximate="true"' in text
    for metric in {
        line.split()[2]
        for line in text.splitlines()
        if line.startswith("# HELP ")
    }:
        assert f"# TYPE {metric} gauge" in text
    assert text.count("# HELP nmesh_telemetry_samples ") == 1


def test_run_returns_failure_when_gateway_is_unavailable(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fail)
    result = cli._run_prompt(SimpleNamespace(prompt="hello", role="chat", json=False))
    assert result == 1


def test_serve_returns_nonzero_for_failed_gateway(monkeypatch) -> None:
    process = SimpleNamespace(pid=123, wait=lambda: 1)
    monkeypatch.setattr(cli, "_launch_gateway", lambda _port, detach: (process, None))
    monkeypatch.setattr(cli, "clear_gateway", lambda _pid: None)
    assert cli._runtime(SimpleNamespace(command="serve", port=18000)) == 1
