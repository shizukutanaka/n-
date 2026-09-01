from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from fastapi.testclient import TestClient

from nmesh import telemetry
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.planner import Policy, build_plan
from nmesh.telemetry import Sample, Telemetry

from .test_planner import profile


def sample(
    key: str = "model|q4|llamacpp|cpu|0",
    service: str = "chat",
    decode_tps: float | None = 10.0,
    ttft_s: float | None = 0.2,
    total_s: float = 1.0,
    at: float = 1.0,
    approximate: bool = True,
) -> Sample:
    return Sample(service, key, decode_tps, ttft_s, total_s, 20, at, approximate)


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
    assert Telemetry(path).samples()[0].approximate is True


def test_bench_overlay_prefers_exact_samples(tmp_path) -> None:
    store = Telemetry(tmp_path / "telemetry.json")
    for value in (10.0, 12.0, 14.0):
        store.record(sample(key="mixed", decode_tps=value))
    for value in (20.0, 22.0, 24.0):
        store.record(sample(key="mixed", decode_tps=value, approximate=False))
    for value in (30.0, 32.0, 34.0):
        store.record(sample(key="only-approx", decode_tps=value))
    assert store.bench_overlay(min_samples=3) == {
        "mixed": 22.0,
        "only-approx": 32.0,
    }


class _TelemetryHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}
    stream: ClassVar[bool] = True
    include_usage: ClassVar[bool] = False

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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index in range(20):
            payload = json.dumps(
                {"choices": [{"delta": {"content": str(index)}}]}
            ).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()
            time.sleep(0.002)
        if self.__class__.include_usage:
            payload = json.dumps({
                "choices": [],
                "usage": {"completion_tokens": 17, "prompt_tokens": 23},
            }).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

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
        _TelemetryHandler.include_usage = False
        upstream.shutdown()
        upstream.server_close()
