from __future__ import annotations

import asyncio
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
from nmesh.runtime import RuntimeStatus
from nmesh.runtime.logs import log_path

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


def _asgi_get(app: object, authorization: bytes) -> int:
    async def invoke() -> int:
        events: list[dict[str, object]] = []
        received = False

        async def receive() -> dict[str, object]:
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            events.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/models",
            "raw_path": b"/v1/models",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"authorization", authorization),
            ],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
        }
        await app(scope, receive, send)
        return next(
            int(message["status"])
            for message in events
            if message["type"] == "http.response.start"
        )

    return asyncio.run(invoke())


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
            assert second.json()["error"]["type"] == "server_error"
            assert second.json()["error"]["code"] == 503
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
        assert response.json()["error"]["type"] == "server_error"
        assert "llamacpp" in response.json()["error"]["message"]
    finally:
        _CompletionHandler.status = 200
        upstream.shutdown()
        upstream.server_close()


def test_cors_preflight_and_response_headers(monkeypatch) -> None:
    monkeypatch.setenv("NMESH_API_KEY", "test-secret")
    plan = _completion_plan(1)
    with TestClient(create_app(plan)) as client:
        preflight = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 204
        assert (
            preflight.headers["Access-Control-Allow-Origin"]
            == "http://localhost:3000"
        )
        assert "authorization" in preflight.headers["Access-Control-Allow-Headers"]
        # Preflight must not be blocked by auth.
        assert "WWW-Authenticate" not in preflight.headers

        models = client.get(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Authorization": "Bearer test-secret",
            },
        )
        assert models.status_code == 200
        assert models.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        # No Origin header -> no CORS headers (non-browser clients unaffected).
        plain = client.get("/v1/models", headers={"Authorization": "Bearer test-secret"})
        assert "Access-Control-Allow-Origin" not in plain.headers


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
        assert missing.json() == {
            "error": {
                "message": "Invalid or missing API key",
                "type": "invalid_request_error",
                "code": 401,
            }
        }
        assert secret not in missing.text
        assert secret not in wrong.text


def test_logs_endpoint_reads_tail_and_reports_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("NMESH_API_KEY", raising=False)
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    path = log_path("chat")
    path.parent.mkdir(parents=True)
    path.write_text("first\nsecond\n", encoding="utf-8")
    plan = _completion_plan(1)

    with TestClient(create_app(plan)) as client:
        response = client.get("/logs/chat?lines=1")
        missing = client.get("/logs/missing")
        traversal = client.get("/logs/..%2F..%2Fplan")

    assert response.status_code == 200
    assert response.json() == {
        "service": "chat",
        "path": str(path),
        "lines": ["second"],
    }
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == 404
    assert traversal.status_code == 404
    assert traversal.json()["error"]["code"] == 404


def test_logs_endpoint_requires_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setenv("NMESH_API_KEY", "test-secret")
    plan = _completion_plan(1)

    with TestClient(create_app(plan)) as client:
        response = client.get("/logs/chat")

    assert response.status_code == 401


def test_gateway_admin_unload_and_running(monkeypatch) -> None:
    plan = _completion_plan(1)
    monkeypatch.setattr(
        gateway_module,
        "runtime_status",
        lambda: SimpleNamespace(services=[{
            "service": plan.services[0].name,
            "running": True,
        }]),
    )
    monkeypatch.setattr(gateway_module, "idle_services", lambda: set())
    unloaded: list[str] = []

    def fake_unload(name: str) -> bool:
        unloaded.append(name)
        return True

    monkeypatch.setattr(gateway_module, "unload", fake_unload)
    with TestClient(create_app(plan)) as client:
        response = client.post("/admin/unload/chat")
        assert response.status_code == 200
        assert response.json() == {
            "unloaded": ["chat"],
            "results": [{"service": "chat", "unloaded": True, "reason": "ok"}],
        }
        unknown = client.post("/admin/unload/missing")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == 404
        running = client.get("/admin/running")
    assert unloaded == ["chat"]
    assert running.status_code == 200
    assert running.json()["services"][0]["in_flight"] == 0


def test_gateway_admin_unload_reports_not_running(monkeypatch) -> None:
    plan = _completion_plan(1)
    monkeypatch.setattr(
        gateway_module,
        "runtime_status",
        lambda: SimpleNamespace(services=[{
            "service": plan.services[0].name,
            "running": False,
        }]),
    )
    monkeypatch.setattr(gateway_module, "idle_services", lambda: set())
    monkeypatch.setattr(gateway_module, "unload", lambda _name: False)

    with TestClient(create_app(plan)) as client:
        response = client.post("/admin/unload/chat")

    assert response.status_code == 200
    assert response.json() == {
        "unloaded": [],
        "results": [{"service": "chat", "unloaded": False, "reason": "not_running"}],
    }


def test_gateway_admin_unload_reports_not_owned(monkeypatch) -> None:
    plan = _completion_plan(1)
    monkeypatch.setattr(
        gateway_module,
        "runtime_status",
        lambda: SimpleNamespace(services=[{
            "service": plan.services[0].name,
            "running": True,
            "shared": False,
            "external": False,
        }]),
    )
    monkeypatch.setattr(gateway_module, "idle_services", lambda: set())
    monkeypatch.setattr(gateway_module, "unload", lambda _name: False)

    with TestClient(create_app(plan)) as client:
        response = client.post("/admin/unload/chat")

    assert response.status_code == 200
    assert response.json() == {
        "unloaded": [],
        "results": [{"service": "chat", "unloaded": False, "reason": "not_owned"}],
    }


def test_gateway_admin_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("NMESH_API_KEY", "test-secret")
    plan = _completion_plan(1)
    with TestClient(create_app(plan)) as client:
        response = client.post("/admin/unload")
    assert response.status_code == 401


def test_status_endpoint_probe_runs_off_the_event_loop(monkeypatch) -> None:
    """runtime_status blocks on per-service health probes — /status must run
    it in a worker thread, not on the event loop."""
    plan = _completion_plan(1)
    probe_threads: list[int] = []

    def probe() -> RuntimeStatus:
        probe_threads.append(threading.get_ident())
        return RuntimeStatus(True, [{"service": "chat", "running": True}])

    monkeypatch.setattr(gateway_module, "runtime_status", probe)
    app = create_app(plan)
    endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", None) == "/status"
    )
    result = asyncio.run(endpoint())
    assert probe_threads
    assert all(tid != threading.get_ident() for tid in probe_threads)
    assert result["services"] == [{"service": "chat", "running": True}]


def test_gateway_reaper_skips_in_flight(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "KEEP_ALIVE", 5.0)
    clock = [100.0]
    monkeypatch.setattr(gateway_module.time, "monotonic", lambda: clock[0])
    plan = _completion_plan(1)
    monkeypatch.setattr(
        gateway_module,
        "runtime_status",
        lambda: SimpleNamespace(services=[{
            "service": "chat",
            "running": True,
        }]),
    )
    unloaded: list[str] = []
    monkeypatch.setattr(gateway_module, "unload", lambda name: unloaded.append(name) or True)
    app = create_app(plan)
    ticket = app.state.in_flight.enter("chat")
    clock[0] = 110.0
    asyncio.run(app.state.reap())
    assert unloaded == []
    app.state.in_flight.leave("chat", ticket)
    asyncio.run(app.state.reap())
    assert unloaded == ["chat"]


def test_openai_model_listing_and_detail() -> None:
    plan = _completion_plan(1)
    with TestClient(create_app(plan)) as client:
        listing = client.get("/v1/models")
        assert listing.status_code == 200
        models = listing.json()["data"]
        assert models
        assert all(isinstance(item["created"], int) for item in models)
        advertised = models[0]["id"]
        detail = client.get(f"/v1/models/{advertised}")
        assert detail.status_code == 200
        assert detail.json() == next(item for item in models if item["id"] == advertised)
        missing = client.get("/v1/models/not-advertised")
        assert missing.status_code == 404
        assert missing.json()["error"]["type"] == "invalid_request_error"
        assert missing.json()["error"]["code"] == 404


def test_reserved_tokens_uses_larger_completion_limit() -> None:
    assert gateway_module._reserved_tokens({"max_completion_tokens": 8}) == 8
    assert gateway_module._reserved_tokens({
        "max_tokens": 4, "max_completion_tokens": 8,
    }) == 8
    assert gateway_module._reserved_tokens({
        "max_tokens": 8, "max_completion_tokens": 4,
    }) == 8
    assert gateway_module._reserved_tokens({
        "max_tokens": None, "max_completion_tokens": "bad",
    }) == 0


def test_non_ascii_api_key_authentication(monkeypatch) -> None:
    secret = "caf\u00e9"
    monkeypatch.setenv("NMESH_API_KEY", secret)
    app = create_app(_completion_plan(1))
    assert _asgi_get(app, b"Bearer wrong") == 401
    assert _asgi_get(app, b"Bearer caf\xc3\xa9") == 200


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
    assert "# HELP nmesh_telemetry_decode_tokens_per_second_median " in text
    assert "nmesh_telemetry_decode_tps_median" not in text
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
    result = cli._run_prompt(
        SimpleNamespace(prompt="hello", role="chat", json=False, port=18000)
    )
    assert result == 1


def test_run_sends_nmesh_api_key_when_set(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, *args, **kwargs):
        captured["authorization"] = request.headers.get("Authorization")
        return Response()

    monkeypatch.setenv("NMESH_API_KEY", "test-key-1")
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    result = cli._run_prompt(SimpleNamespace(prompt="hi", role="chat", json=False))
    assert result == 0
    assert captured["authorization"] == "Bearer test-key-1"


def test_gateway_headers_empty_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("NMESH_API_KEY", raising=False)
    assert cli._gateway_headers() == {}


def test_run_surfaces_upstream_error_body(monkeypatch, capsys) -> None:
    import io
    import urllib.error

    def fail(*args, **kwargs):
        body = json.dumps(
            {"error": {"message": "the current context does not logits computation"}}
        ).encode()
        raise urllib.error.HTTPError(
            "http://127.0.0.1:18000/v1/chat/completions",
            500, "Internal Server Error", {}, io.BytesIO(body),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fail)
    result = cli._run_prompt(SimpleNamespace(prompt="hi", role="embed", json=False))
    assert result == 1
    err = capsys.readouterr().err
    assert "HTTP 500" in err
    assert "logits computation" in err


def test_serve_returns_nonzero_for_failed_gateway(monkeypatch) -> None:
    process = SimpleNamespace(pid=123, wait=lambda: 1)
    monkeypatch.setattr(cli, "_launch_gateway", lambda _port, detach: (process, None))
    monkeypatch.setattr(cli, "clear_gateway", lambda _pid: None)
    assert cli._runtime(SimpleNamespace(command="serve", port=18000)) == 1
