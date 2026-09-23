from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

from fastapi.testclient import TestClient

import nmesh.gateway as gateway_module
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.gateway.limit import SlotLimiter
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _llama_plan(slots: int):
    model = ModelSpec(
        "limited-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        memory=replace(plan.services[0].memory, parallel_slots=slots),
    )
    return replace(plan, services=[service])


class _LimitHandler(BaseHTTPRequestHandler):
    requests = 0
    status = 200
    block = False
    stream = False
    slots: ClassVar[list[object] | None] = None
    started: ClassVar[threading.Event] = threading.Event()
    second_started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def do_GET(self) -> None:
        if self.path == "/slots" and type(self).slots is not None:
            payload = json.dumps(type(self).slots).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests += 1
        if type(self).requests >= 2:
            type(self).second_started.set()
        type(self).started.set()
        if type(self).stream and body.get("stream"):
            self.send_response(type(self).status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
            self.wfile.flush()
            type(self).release.wait(timeout=5)
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if type(self).block:
            type(self).release.wait(timeout=5)
        payload = json.dumps({"model": "test", "choices": []}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _limit_upstream() -> ThreadingHTTPServer:
    _LimitHandler.requests = 0
    _LimitHandler.status = 200
    _LimitHandler.block = False
    _LimitHandler.stream = False
    _LimitHandler.slots = None
    _LimitHandler.started.clear()
    _LimitHandler.second_started.clear()
    _LimitHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _LimitHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    return upstream


def _reserved_port() -> socket.socket:
    reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reserved.bind(("127.0.0.1", 0))
    return reserved


def test_slot_limiter_enforces_limits_and_releases() -> None:
    plan = _llama_plan(1)
    service = plan.services[0]

    async def scenario() -> None:
        limiter = SlotLimiter()
        limiter.size(plan)
        first = await limiter.acquire(service, 1.0)
        assert first is not None
        waiting = asyncio.create_task(limiter.acquire(service, 0.01))
        await asyncio.sleep(0)
        assert limiter.metrics()[service.name]["waiting"] == 1
        assert await waiting is None
        assert limiter.metrics()[service.name]["in_flight"] == 1
        limiter.release(first)
        second = await limiter.acquire(service, 1.0)
        assert second is not None
        limiter.release(second)
        assert limiter.metrics()[service.name]["in_flight"] == 0

    asyncio.run(scenario())


def test_slot_limiter_releases_tokens_from_old_plan_after_resize() -> None:
    old = _llama_plan(1)
    new = _llama_plan(2)

    async def scenario() -> None:
        limiter = SlotLimiter()
        limiter.size(old)
        token = await limiter.acquire(old.services[0], 1.0)
        assert token is not None
        limiter.size(new)
        limiter.release(token)
        assert limiter.metrics()[new.services[0].name]["in_flight"] == 0
        first = await limiter.acquire(new.services[0], 1.0)
        second = await limiter.acquire(new.services[0], 1.0)
        assert first is not None and second is not None
        limiter.release(first)
        limiter.release(second)

    asyncio.run(scenario())


def test_gateway_metrics_report_limited_services_only() -> None:
    limited = _llama_plan(2)
    with TestClient(create_app(limited)) as client:
        metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["concurrency"] == {
        "chat": {"limit": 2, "in_flight": 0, "waiting": 0}
    }

    model = ModelSpec(
        "ollama-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"ollama": "test:latest"},
    )
    unlimited = build_plan(profile(8), [model], Policy(roles=["chat"]))
    with TestClient(create_app(unlimited)) as client:
        metrics = client.get("/metrics")
    assert metrics.json()["concurrency"] == {}


def test_embeddings_skip_slot_limiter(monkeypatch) -> None:
    model = ModelSpec(
        "embed-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["embed"]))
    called = False

    async def fail_acquire(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("embeddings must not acquire a slot")

    monkeypatch.setattr(gateway_module.SlotLimiter, "acquire", fail_acquire)
    # Bound-but-not-listening ports wait out CONNECT_TIMEOUT before failing.
    monkeypatch.setattr(gateway_module, "CONNECT_TIMEOUT", 0.2)
    with _reserved_port() as reserved:
        service = replace(plan.services[0], port=reserved.getsockname()[1])
        isolated_plan = replace(plan, services=[service])
        with TestClient(create_app(isolated_plan)) as client:
            response = client.post("/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 502
    assert not called


def test_gateway_slot_timeout_returns_retryable_503(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        monkeypatch.setattr(gateway_module, "QUEUE_TIMEOUT", 0.01)
        with TestClient(create_app(plan)) as client:
            first_result: list[object] = []
            first = threading.Thread(target=lambda: first_result.append(
                client.post("/v1/chat/completions", json={"messages": []})
            ))
            _LimitHandler.block = True
            first.start()
            assert _LimitHandler.started.wait(timeout=2)
            second = client.post("/v1/chat/completions", json={"messages": []})
            assert second.status_code == 503
            assert second.headers["Retry-After"] == "1"
            assert second.json()["error"]["type"] == "server_error"
            assert "1 slots" in second.json()["error"]["message"]
            _LimitHandler.release.set()
            first.join(timeout=5)
            assert first_result[0].status_code == 200
            _LimitHandler.block = False
            assert client.post(
                "/v1/chat/completions", json={"messages": []}
            ).status_code == 200
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_gateway_upstream_error_releases_slot(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        _LimitHandler.status = 500
        with TestClient(create_app(plan)) as client:
            failed = client.post("/v1/chat/completions", json={"messages": []})
            assert failed.status_code == 500
            assert client.get("/metrics").json()["concurrency"]["chat"] == {
                "limit": 1, "in_flight": 0, "waiting": 0
            }
            _LimitHandler.status = 200
            assert client.post(
                "/v1/chat/completions", json={"messages": []}
            ).status_code == 200
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_unlimited_ollama_requests_are_not_serialized(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        model = ModelSpec(
            "ollama-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
            4096, ["chat"], 80.0, "test", {"ollama": "test:latest"},
        )
        plan = build_plan(profile(8), [model], Policy(roles=["chat"]))
        monkeypatch.setattr(
            gateway_module, "_base_url",
            lambda _service: f"http://127.0.0.1:{upstream.server_address[1]}",
        )
        _LimitHandler.block = True
        with TestClient(create_app(plan)) as client:
            results: list[object] = []
            threads = [
                threading.Thread(target=lambda: results.append(
                    client.post("/v1/chat/completions", json={"messages": []})
                ))
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            assert _LimitHandler.second_started.wait(timeout=2)
            _LimitHandler.release.set()
            for thread in threads:
                thread.join(timeout=5)
            assert len(results) == 2
            assert all(response.status_code == 200 for response in results)
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_jobs_endpoint_tracks_request_lifecycle() -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        with TestClient(create_app(plan)) as client:
            response = client.post("/v1/chat/completions", json={"messages": []})
            assert response.status_code == 200
            job_id = response.headers["x-nmesh-job-id"]
            job = client.get(f"/v1/jobs/{job_id}").json()
            assert job["state"] == "done"
            assert job["service"] == "chat"
            listing = client.get("/v1/jobs").json()
            assert any(item["id"] == job_id for item in listing["jobs"])
            assert client.get("/v1/jobs/job-nope").status_code == 404
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_jobs_report_decode_progress_from_llamacpp_slots(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        # /slots reports next_token as a one-element list.
        _LimitHandler.slots = [
            {"is_processing": True,
             "next_token": [{"n_decoded": 12, "n_remain": 88}]},
        ]
        monkeypatch.setattr(
            gateway_module, "_service_is_running_llamacpp", lambda _s: True
        )
        with TestClient(create_app(plan)) as client:
            _LimitHandler.block = True
            thread = threading.Thread(
                target=lambda: client.post(
                    "/v1/chat/completions", json={"messages": []}
                )
            )
            thread.start()
            assert _LimitHandler.started.wait(timeout=2)
            try:
                jobs = client.get("/v1/jobs").json()["jobs"]
                running = next(j for j in jobs if j["state"] == "running")
                assert running["progress"] == {"decoded": 12, "remaining": 88}
                single = client.get(f"/v1/jobs/{running['id']}").json()
                assert single["progress"]["decoded"] == 12
            finally:
                _LimitHandler.release.set()
                thread.join(timeout=5)
                _LimitHandler.block = False
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_jobs_omit_progress_when_slot_mapping_is_ambiguous(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        # Two processing slots cannot be mapped to a single running job.
        _LimitHandler.slots = [
            {"is_processing": True, "next_token": [{"n_decoded": 1}]},
            {"is_processing": True, "next_token": [{"n_decoded": 2}]},
        ]
        monkeypatch.setattr(
            gateway_module, "_service_is_running_llamacpp", lambda _s: True
        )
        with TestClient(create_app(plan)) as client:
            _LimitHandler.block = True
            thread = threading.Thread(
                target=lambda: client.post(
                    "/v1/chat/completions", json={"messages": []}
                )
            )
            thread.start()
            assert _LimitHandler.started.wait(timeout=2)
            try:
                jobs = client.get("/v1/jobs").json()["jobs"]
                running = next(j for j in jobs if j["state"] == "running")
                assert "progress" not in running
            finally:
                _LimitHandler.release.set()
                thread.join(timeout=5)
                _LimitHandler.block = False
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_jobs_show_queued_state_and_queue_timeout_failure(monkeypatch) -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        monkeypatch.setattr(gateway_module, "QUEUE_TIMEOUT", 0.05)
        with TestClient(create_app(plan)) as client:
            _LimitHandler.block = True
            first = threading.Thread(
                target=lambda: client.post(
                    "/v1/chat/completions", json={"messages": []}
                )
            )
            first.start()
            assert _LimitHandler.started.wait(timeout=2)
            queued = client.get("/v1/jobs").json()
            assert queued["counts"]["chat"]["running"] == 1
            timed_out = client.post(
                "/v1/chat/completions", json={"messages": []}
            )
            assert timed_out.status_code == 503
            assert "queue position" in timed_out.json()["error"]["message"]
            jobs = client.get("/v1/jobs").json()["jobs"]
            failed = next(
                job for job in jobs if job["state"] == "failed"
            )
            assert failed["detail"] == "queue_timeout"
            _LimitHandler.release.set()
            first.join(timeout=5)
            _LimitHandler.block = False
    finally:
        upstream.shutdown()
        upstream.server_close()
def test_job_registry_transitions_and_capacity() -> None:
    from nmesh.gateway.jobs import JobRegistry

    registry = JobRegistry(capacity=2)
    first = registry.submit("chat", "/v1/chat/completions")
    second = registry.submit("chat", "/v1/chat/completions")
    assert registry.position(second) == 2
    registry.start(first)
    assert registry.position(first) == 0
    registry.finish(first, ok=True)
    third = registry.submit("embed", "/v1/embeddings")
    registry.finish(second, ok=False, detail="boom")
    registry.finish(third, ok=True)
    # capacity keeps only the 2 most recent finished jobs
    assert registry.get(first.id) is None
    assert registry.get(third.id).state == "done"
    counts = registry.counts()
    assert counts == {}
    running = registry.submit("chat", "/v1/chat/completions")
    registry.start(running)
    assert registry.counts() == {"chat": {"queued": 0, "running": 1}}
    # finish is idempotent
    registry.finish(running, ok=True)
    registry.finish(running, ok=False)
    assert registry.get(running.id).state == "done"


def test_jobs_cancel_queued_job() -> None:
    upstream = _limit_upstream()
    try:
        plan = _llama_plan(1)
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        with TestClient(create_app(plan)) as client:
            _LimitHandler.block = True
            first = threading.Thread(
                target=lambda: client.post(
                    "/v1/chat/completions", json={"messages": []}
                )
            )
            first.start()
            assert _LimitHandler.started.wait(timeout=2)
            results: list[object] = []
            second = threading.Thread(
                target=lambda: results.append(
                    client.post("/v1/chat/completions", json={"messages": []})
                )
            )
            second.start()
            queued_id = None
            for _ in range(50):
                queued = [
                    j for j in client.get("/v1/jobs").json()["jobs"]
                    if j["state"] == "queued"
                ]
                if queued:
                    queued_id = queued[0]["id"]
                    break
                time.sleep(0.05)
            assert queued_id is not None
            cancelled = client.delete(f"/v1/jobs/{queued_id}")
            assert cancelled.status_code == 200
            assert cancelled.json()["state"] == "cancelled"
            second.join(timeout=5)
            assert results[0].status_code == 409
            again = client.delete(f"/v1/jobs/{queued_id}")
            assert again.status_code == 409
            assert client.delete("/v1/jobs/job-nope").status_code == 404
            _LimitHandler.release.set()
            first.join(timeout=5)
            _LimitHandler.block = False
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_job_registry_cancel() -> None:
    from nmesh.gateway.jobs import JobRegistry

    registry = JobRegistry()
    queued = registry.submit("chat", "/v1/chat/completions")
    assert registry.cancel(queued) is True
    assert registry.get(queued.id).state == "cancelled"
    assert registry.cancel(queued) is False
    running = registry.submit("chat", "/v1/chat/completions")
    registry.start(running)
    assert registry.cancel(running) is False
    listed = [j.id for j in registry.list()]
    assert queued.id in listed and running.id in listed


def test_slot_progress_polls_services_concurrently(monkeypatch) -> None:
    """Two running jobs on two services are polled in parallel — elapsed
    tracks one 0.8s timeout budget, not the sum across services."""
    plan = _llama_plan(1)
    svc_a = replace(plan.services[0], name="svc-a")
    svc_b = replace(plan.services[0], name="svc-b")
    job_a = SimpleNamespace(id="job-a", service="svc-a", state="running")
    job_b = SimpleNamespace(id="job-b", service="svc-b", state="running")

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

        async def get(self, path: str) -> object:
            await asyncio.sleep(0.3)
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"is_processing": True,
                     "next_token": [{"n_decoded": 5}]},
                ],
            )

    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        gateway_module, "_service_is_running_llamacpp", lambda _s: True
    )
    monkeypatch.setattr(gateway_module, "_base_url", lambda _s: "http://x")

    async def scenario() -> tuple[float, dict[str, dict[str, int]]]:
        started = time.monotonic()
        result = await gateway_module._slot_progress(
            [svc_a, svc_b], [job_a, job_b]
        )
        return time.monotonic() - started, result

    elapsed, progress = asyncio.run(scenario())
    assert elapsed < 0.6
    assert progress == {
        "job-a": {"decoded": 5},
        "job-b": {"decoded": 5},
    }
