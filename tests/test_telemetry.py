from __future__ import annotations

import inspect
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

import nmesh.bench.runner as bench_runner
from nmesh import telemetry
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.planner import Policy, build_plan
from nmesh.telemetry import (
    COMPARABLE_PROMPT_TOKENS,
    Sample,
    Telemetry,
)

from .test_planner import profile


def sample(
    key: str = "model|q4|llamacpp|cpu|0",
    service: str = "chat",
    decode_tps: float | None = 10.0,
    ttft_s: float | None = 0.2,
    total_s: float = 1.0,
    at: float = 1.0,
    approximate: bool = True,
    prefill_tps: float | None = None,
    in_flight: int | None = 1,
    prompt_tokens: int | None = 512,
) -> Sample:
    return Sample(service, key, decode_tps, ttft_s, total_s, 20, at, approximate,
                  prefill_tps, in_flight, prompt_tokens)


def test_record_round_trip_and_trim(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json", max_samples=2)
    store.record(sample(at=1))
    store.record(sample(at=2))
    store.record(sample(at=3))
    store.record(sample(key="other", at=4))
    assert [item.at for item in store.samples()] == [2, 3, 4]
    assert json.loads((tmp_path / "telemetry.json").read_text())["samples"]


def test_summary_medians_and_p95(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for index, ttft in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        store.record(sample(decode_tps=float(index + 1), ttft_s=ttft,
                            total_s=float(index + 2), at=float(index)))
    store.record(sample(decode_tps=None, ttft_s=None, total_s=9.0, at=10))
    result = store.summary()["chat"]
    assert result["samples"] == 6
    assert result["decode_tps_median"] == 3
    assert result["ttft_s_median"] == 0.3
    assert result["ttft_s_p95"] == 0.5
    assert result["total_s_median"] == 4.5


def test_summary_includes_prefill_median(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    store.record(sample(prefill_tps=400.0))
    store.record(sample(prefill_tps=420.0))
    assert store.summary()["chat"]["prefill_tps_median"] == 410.0


def test_bench_overlay_min_samples_and_ignores_missing_decode(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for value in (10.0, 12.0, 14.0):
        store.record(sample(decode_tps=value))
    store.record(sample(decode_tps=None))
    store.record(sample(key="few", decode_tps=20.0))
    assert store.bench_overlay(min_samples=3) == {
        "model|q4|llamacpp|cpu|0": 12.0,
    }


def test_corrupt_json_is_empty(tmp_path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text("{not json", encoding="utf-8")
    assert Telemetry(path).samples() == []


def test_old_telemetry_samples_default_to_approximate(tmp_path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps({"samples": [{
        "service": "chat",
        "key": "old",
        "decode_tps": 10,
        "ttft_s": 0.2,
        "total_s": 1,
        "completion_tokens": 20,
        "at": 1,
    }]}), encoding="utf-8")
    loaded = Telemetry(path).samples()[0]
    assert loaded.approximate is True
    assert loaded.prefill_tps is None
    assert loaded.in_flight is None
    assert loaded.prompt_tokens is None
    assert Telemetry(path).bench_overlay() == {}


def test_bench_overlay_prefers_exact_samples(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for value in (10.0, 12.0, 14.0):
        store.record(sample(key="mixed", decode_tps=value))
    for value in (20.0, 22.0, 24.0):
        store.record(sample(key="mixed", decode_tps=value, approximate=False))
    for value in (30.0, 32.0, 34.0):
        store.record(sample(key="only-approx", decode_tps=value))
    for value in (1.0, 3.0):
        store.record(sample(key="fallback", decode_tps=value, approximate=False))
    for value in range(10, 20):
        store.record(sample(key="fallback", decode_tps=float(value)))
    assert store.bench_overlay(min_samples=3) == {
        "mixed": 22.0,
        "only-approx": 32.0,
        "fallback": 14.5,
    }


def test_bench_overlay_uses_single_stream_samples_and_reports_skips(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for value in (10.0, 12.0, 14.0):
        store.record(sample(key="single", decode_tps=value, in_flight=1))
    for value in (30.0, 32.0, 34.0):
        store.record(sample(key="busy", decode_tps=value, in_flight=4))
    assert store.bench_overlay(min_samples=3) == {"single": 12.0}
    report = store.overlay_report(min_samples=3)
    assert report.values == {"single": 12.0}
    assert report.under_load == 3


@pytest.mark.parametrize("prompt_tokens", [1025, 2005, 8192])
def test_overlay_excludes_deep_samples_and_counts_them(tmp_path, prompt_tokens) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    # 2005 real prompt tokens measured 0.883 of the reference decode rate.
    for value in (34.20, 35.0, 33.5):
        store.record(sample(key="deep", decode_tps=value, prompt_tokens=prompt_tokens))
    report = store.overlay_report(min_samples=3)
    assert report.values == {}
    assert report.off_reference == 3


@pytest.mark.parametrize("prompt_tokens", [256, 1024])
def test_overlay_accepts_comparable_prompt_depths(tmp_path, prompt_tokens) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    store.record(sample(key="eligible", decode_tps=45.77, prompt_tokens=prompt_tokens))
    report = store.overlay_report(min_samples=1)
    assert report.values == {"eligible": 45.77}
    assert report.off_reference == 0


def test_overlay_mixed_depth_fixture_uses_shallow_measured_median(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for _ in range(3):
        store.record(sample(key="mixed-depth", decode_tps=45.77, prompt_tokens=256))
    for _ in range(3):
        store.record(sample(key="mixed-depth", decode_tps=34.20, prompt_tokens=8192))
    report = store.overlay_report(min_samples=3)
    # Measured 256-token requests were 45.77 tok/s; 8192-token requests were 34.20.
    assert report.values == {"mixed-depth": 45.77}
    assert report.off_reference == 3


def test_overlay_excludes_unknown_prompt_depth_and_counts_it(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    store.record(sample(key="unknown", decode_tps=20.0, prompt_tokens=None))
    report = store.overlay_report(min_samples=1)
    assert report.values == {}
    assert report.unknown_depth == 1


def test_prompt_depth_round_trip(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    store.record(sample(prompt_tokens=123))
    assert store.samples()[0].prompt_tokens == 123


def test_comparable_depth_exceeds_bench_nominal_default() -> None:
    default = inspect.signature(bench_runner.measure).parameters["prefill_tokens"].default
    # The nominal 512 is below 1024, and its roughly 336 real tokens are lower still.
    assert default < COMPARABLE_PROMPT_TOKENS


class _TelemetryHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}
    stream: ClassVar[bool] = True
    include_usage: ClassVar[bool] = False
    usage_on_every_chunk: ClassVar[bool] = False
    chunks: ClassVar[int] = 20
    completion_tokens: ClassVar[int] = 17
    timings: ClassVar[dict[str, object] | None] = None

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        if not self.stream:
            payload = json.dumps({
                "id": "completion", "object": "chat.completion",
                "model": self.request_body["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"completion_tokens": 7},
            }).encode()
            if self.__class__.timings is not None:
                payload = json.dumps({
                    "id": "completion", "object": "chat.completion",
                    "model": self.request_body["model"],
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"completion_tokens": 7},
                    "timings": self.__class__.timings,
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index in range(self.__class__.chunks):
            data: dict[str, object] = {
                "choices": [{"delta": {"content": str(index)}}]
            }
            if self.__class__.usage_on_every_chunk:
                data["usage"] = {
                    "completion_tokens": index + 1,
                    "prompt_tokens": 23,
                }
            payload = json.dumps(data).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()
            time.sleep(0.002)
        if self.__class__.include_usage or self.__class__.timings is not None:
            final: dict[str, object] = {
                "choices": [],
                "usage": {
                    "completion_tokens": self.__class__.completion_tokens,
                    "prompt_tokens": 23,
                },
            }
            if self.__class__.timings is not None:
                final["timings"] = self.__class__.timings
            payload = json.dumps(final).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


class _ConcurrentHandler(BaseHTTPRequestHandler):
    wait_for_pair: ClassVar[bool] = False
    barrier: ClassVar[threading.Barrier | None] = None

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        if self.__class__.wait_for_pair and self.__class__.barrier is not None:
            self.__class__.barrier.wait(timeout=5)
        body = json.dumps({
            "model": "telemetry-model",
            "choices": [],
            "usage": {"completion_tokens": 1},
            "timings": {"predicted_n": 1, "predicted_ms": 100.0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _gateway_plan(port: int):
    model = ModelSpec("telemetry-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    return replace(plan, services=[replace(plan.services[0], port=port)])


def test_gateway_records_stream_and_nonstream_telemetry(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.include_usage = False
        plan = _gateway_plan(upstream.server_address[1])
        client = TestClient(create_app(plan))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        stream_sample = store.samples()[-1]
        assert stream_sample.decode_tps is not None and stream_sample.decode_tps > 0
        assert stream_sample.ttft_s is not None and stream_sample.ttft_s > 0
        assert stream_sample.approximate is True
        _TelemetryHandler.stream = False
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto", "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        nonstream_sample = store.samples()[-1]
        assert nonstream_sample.decode_tps is None
        assert nonstream_sample.completion_tokens == 7
        assert client.get("/metrics").json()["services"]["chat"]["samples"] == 2
        assert not (tmp_path / "bench.json").exists()
    finally:
        _TelemetryHandler.timings = None
        upstream.shutdown()
        upstream.server_close()


def test_gateway_records_peak_in_flight_for_overlapping_nonstream_requests(
    tmp_path, monkeypatch
) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ConcurrentHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _ConcurrentHandler.wait_for_pair = False
        client = TestClient(create_app(_gateway_plan(upstream.server_address[1])))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert store.samples()[-1].in_flight == 1

        _ConcurrentHandler.barrier = threading.Barrier(2)
        _ConcurrentHandler.wait_for_pair = True
        responses: list[object] = []

        def request() -> None:
            responses.append(client.post("/v1/chat/completions", json={
                "model": "nmesh-auto",
                "stream": False,
                "messages": [{"role": "user", "content": "hello"}],
            }))

        workers = [threading.Thread(target=request) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)
        assert len(responses) == 2
        assert all(response.status_code == 200 for response in responses)
        assert [sample.in_flight for sample in store.samples()[-2:]] == [2, 2]
    finally:
        _ConcurrentHandler.wait_for_pair = False
        _ConcurrentHandler.barrier = None
        upstream.shutdown()
        upstream.server_close()


def test_gateway_records_upstream_stream_timings(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        _TelemetryHandler.timings = {
            "cache_n": 0,
            "prompt_n": 330,
            "prompt_ms": 816.236,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        client = TestClient(create_app(_gateway_plan(upstream.server_address[1])))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        sample = store.samples()[-1]
        assert sample.decode_tps == pytest.approx(41.92, rel=1e-3)
        assert sample.prefill_tps == pytest.approx(404.29, rel=1e-3)
        assert sample.approximate is False
        assert "nmesh_telemetry_prefill_tokens_per_second_median" in (
            client.get("/metrics/prometheus").text
        )
    finally:
        _TelemetryHandler.timings = None
        _TelemetryHandler.include_usage = False
        upstream.shutdown()
        upstream.server_close()


def test_gateway_keeps_partially_cached_prefill_timing(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        _TelemetryHandler.timings = {
            "cache_n": 24,
            "prompt_n": 314,
            "prompt_ms": 816.236,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        client = TestClient(create_app(_gateway_plan(upstream.server_address[1])))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert store.samples()[-1].prefill_tps == pytest.approx(384.9, rel=1e-3)
    finally:
        _TelemetryHandler.timings = None
        _TelemetryHandler.include_usage = False
        upstream.shutdown()
        upstream.server_close()


def test_gateway_omits_cached_prefill_timing(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        _TelemetryHandler.timings = {
            "cache_n": 329,
            "prompt_n": 1,
            "prompt_ms": 3.0,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        client = TestClient(create_app(_gateway_plan(upstream.server_address[1])))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert store.samples()[-1].prefill_tps is None
    finally:
        _TelemetryHandler.timings = None
        _TelemetryHandler.include_usage = False
        upstream.shutdown()
        upstream.server_close()


def test_gateway_records_upstream_nonstream_timings(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = False
        _TelemetryHandler.timings = {
            "cache_n": 0,
            "prompt_n": 330,
            "prompt_ms": 816.236,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        client = TestClient(create_app(_gateway_plan(upstream.server_address[1])))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        sample = store.samples()[-1]
        assert sample.decode_tps == pytest.approx(41.92, rel=1e-3)
        assert sample.prefill_tps == pytest.approx(404.29, rel=1e-3)
        assert sample.ttft_s is None
    finally:
        _TelemetryHandler.stream = True
        _TelemetryHandler.timings = None
        upstream.shutdown()
        upstream.server_close()


def test_gateway_uses_client_requested_usage_without_mutating_request(
    tmp_path, monkeypatch
) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        plan = _gateway_plan(upstream.server_address[1])
        client = TestClient(create_app(plan))
        request = {
            "model": "nmesh-auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        }
        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        assert _TelemetryHandler.request_body == request | {
            "model": plan.services[0].model_ref,
        }
        sample = store.samples()[-1]
        assert sample.approximate is False
        assert sample.completion_tokens == 17
    finally:
        _TelemetryHandler.chunks = 20
        _TelemetryHandler.completion_tokens = 17
        _TelemetryHandler.include_usage = False
        _TelemetryHandler.usage_on_every_chunk = False
        upstream.shutdown()
        upstream.server_close()


def test_gateway_keeps_usage_bearing_token_chunks(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        _TelemetryHandler.usage_on_every_chunk = True
        plan = _gateway_plan(upstream.server_address[1])
        client = TestClient(create_app(plan))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        sample = store.samples()[-1]
        assert sample.approximate is False
        assert sample.completion_tokens == 17
        assert sample.ttft_s is not None
        assert sample.decode_tps is not None
    finally:
        _TelemetryHandler.include_usage = False
        _TelemetryHandler.usage_on_every_chunk = False
        upstream.shutdown()
        upstream.server_close()


def test_gateway_short_stream_records_no_decode_rate(tmp_path, monkeypatch) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "_default", store)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _TelemetryHandler.stream = True
        _TelemetryHandler.include_usage = True
        _TelemetryHandler.chunks = 5
        _TelemetryHandler.completion_tokens = 5
        plan = _gateway_plan(upstream.server_address[1])
        client = TestClient(create_app(plan))
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        sample = store.samples()[-1]
        assert sample.decode_tps is None
        assert sample.ttft_s is not None
    finally:
        _TelemetryHandler.chunks = 20
        _TelemetryHandler.completion_tokens = 17
        _TelemetryHandler.include_usage = False
        upstream.shutdown()
        upstream.server_close()
