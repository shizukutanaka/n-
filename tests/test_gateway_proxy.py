from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import httpx
import pytest
from fastapi.testclient import TestClient

import nmesh.bench.runner as bench_runner
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
        if not self.__class__.request_body.get("stream"):
            body = b'{"model":"upstream-model","choices":[],"usage":{"completion_tokens":1}}'
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
    request_bodies: ClassVar[list[dict[str, object]]] = []
    stream: ClassVar[bool] = True
    usage_on_every_chunk: ClassVar[bool] = False
    timings: ClassVar[dict[str, object] | None] = None
    cached_tokens: ClassVar[int | None] = None
    prompt_tokens: ClassVar[int | None] = 23

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        self.__class__.request_bodies.append(self.__class__.request_body)
        if not self.__class__.stream:
            usage: dict[str, object] = {"completion_tokens": 17}
            if self.__class__.prompt_tokens is not None:
                usage["prompt_tokens"] = self.__class__.prompt_tokens
            payload_data: dict[str, object] = {
                "model": self.request_body["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": usage,
            }
            if self.__class__.timings is not None:
                payload_data["timings"] = self.__class__.timings
            payload = json.dumps(payload_data).encode()
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
            data: dict[str, object] = {
                "choices": [{"delta": {"content": str(index)}}],
            }
            if self.__class__.usage_on_every_chunk:
                usage: dict[str, object] = {"completion_tokens": index + 1}
                if self.__class__.prompt_tokens is not None:
                    usage["prompt_tokens"] = self.__class__.prompt_tokens
                data["usage"] = usage
            payload = json.dumps(data).encode()
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()
        usage = {"completion_tokens": 17}
        if self.__class__.prompt_tokens is not None:
            usage["prompt_tokens"] = self.__class__.prompt_tokens
        if self.__class__.cached_tokens is not None:
            usage["prompt_tokens_details"] = {
                "cached_tokens": self.__class__.cached_tokens,
            }
        final: dict[str, object] = {
            "choices": [],
            "usage": usage,
        }
        if self.__class__.timings is not None:
            final["timings"] = self.__class__.timings
        payload = json.dumps(final).encode()
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


class _RerankHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        payload = json.dumps({
            "model": "/internal/path/model.gguf",
            "object": "list",
            "results": [{"index": 0, "relevance_score": 0.5}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_gateway_rerank_proxies_to_rerank_service() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _RerankHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("embed-model", "test", 500_000_000, 24, 16, 2, 64,
                          1024, 4096, ["embed"], 80.0, "test",
                          {"hf_gguf": "test/repo"})
        plan = build_plan(
            profile(64, (24,)), [model], Policy(roles=["rerank"]),
        )
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        client = TestClient(create_app(plan))
        response = client.post("/v1/rerank", json={
            "model": "nmesh-auto", "query": "q", "documents": ["a", "b"],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["relevance_score"] == 0.5
        assert body["model"] == "nmesh-auto"
        assert _RerankHandler.request_body["model"] == service.model_ref
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_gateway_rerank_501_without_rerank_service() -> None:
    model = ModelSpec("embed-model", "test", 500_000_000, 24, 16, 2, 64,
                      1024, 4096, ["embed"], 80.0, "test",
                      {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["embed"]))
    client = TestClient(create_app(plan))
    response = client.post("/v1/rerank", json={
        "model": "nmesh-auto", "query": "q", "documents": ["a", "b"],
    })
    assert response.status_code == 501
    assert "rerank" in response.text


class _AnthropicHandler(BaseHTTPRequestHandler):
    request_body: ClassVar[dict[str, object]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.request_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        if self.__class__.request_body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in (
                b"event: message_start\n"
                + b'data: {"type":"message_start","message":{"model":"upstream-model","usage":{"input_tokens":9}}}\n\n',
                b"event: content_block_delta\n"
                + b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n',
                b"event: message_stop\n" + b'data: {"type":"message_stop"}\n\n',
            ):
                self.wfile.write(frame)
                self.wfile.flush()
            return
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({
                "id": "msg_1", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": self.request_body["model"],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 2},
            }).encode()
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def test_gateway_forwards_anthropic_messages_without_openai_fields() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("proxy-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        plan = replace(plan, services=[service])
        client = TestClient(create_app(plan))
        response = client.post("/v1/messages", json={
            "model": "nmesh-auto", "stream": True, "max_tokens": 16,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": "hello"}]}],
        })
        assert response.status_code == 200
        # Anthropic SSE frames (event: lines) pass through untouched, and the
        # upstream model id nested in message_start is rewritten to the
        # client-facing model just like a non-streamed response.
        assert "event: message_start" in response.text
        assert '"model":"nmesh-auto"' in response.text
        assert "upstream-model" not in response.text
        assert _AnthropicHandler.request_body["model"] == service.model_ref
        # stream_options is an OpenAI-only field; it must not be injected
        # into an Anthropic request even when the client asked to stream.
        assert "stream_options" not in _AnthropicHandler.request_body
    finally:
        upstream.shutdown()
        upstream.server_close()


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


def test_gateway_defaults_omitted_model_to_catalog_id_for_all_responses() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CompatStreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec("compat-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        client = TestClient(create_app(replace(plan, services=[service])))
        stream = client.post("/v1/chat/completions", json={
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        nonstream = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert stream.status_code == 200
        assert nonstream.status_code == 200
        assert service.model_ref not in stream.text
        assert service.model_ref not in nonstream.text
        assert all(
            payload["model"] == service.model_id
            for line in stream.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
            and line != "data: {not-json}"
            for payload in [json.loads(line[6:])]
        )
        assert nonstream.json()["model"] == service.model_id
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


def test_gateway_telemetry_records_prompt_depth_from_stream_and_nonstream_usage(
    monkeypatch,
) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    samples = []
    monkeypatch.setattr(gateway_module, "record_telemetry", samples.append)
    try:
        _UsageHandler.stream = True
        _UsageHandler.prompt_tokens = 23
        _UsageHandler.timings = None
        model = ModelSpec("depth-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        service = replace(plan.services[0], port=upstream.server_address[1])
        client = TestClient(create_app(replace(plan, services=[service])))

        streamed = client.post("/v1/chat/completions", json={
            "model": "client-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert streamed.status_code == 200
        assert samples[-1].prompt_tokens == 23

        _UsageHandler.stream = False
        nonstreamed = client.post("/v1/chat/completions", json={
            "model": "client-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert nonstreamed.status_code == 200
        assert samples[-1].prompt_tokens == 23
    finally:
        _UsageHandler.stream = True
        _UsageHandler.prompt_tokens = 23
        _UsageHandler.timings = None
        upstream.shutdown()
        upstream.server_close()


def test_gateway_telemetry_derives_prompt_depth_from_llamacpp_timings(
    monkeypatch,
) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    samples = []
    monkeypatch.setattr(gateway_module, "record_telemetry", samples.append)
    try:
        _UsageHandler.stream = True
        _UsageHandler.prompt_tokens = None
        _UsageHandler.timings = {
            "prompt_n": 314,
            "cache_n": 24,
            "prompt_ms": 816.236,
        }
        model = ModelSpec("timing-depth-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
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
        assert samples[-1].prompt_tokens == 338
    finally:
        _UsageHandler.stream = True
        _UsageHandler.prompt_tokens = 23
        _UsageHandler.timings = None
        upstream.shutdown()
        upstream.server_close()


def test_non_usage_backend_stream_does_not_inject_usage(monkeypatch) -> None:
    """Backends outside _STREAM_USAGE_BACKENDS (mlx: stream_options support
    unverified upstream) must receive the request body untouched."""
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
        for backend in ("mlx",):
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
                         prefill_tokens=8, decode_tokens=2, runs=1)
        assert result.prefill_tps > 0
        assert result.decode_tps > 0
        assert result.decode_tps_min == result.decode_tps
        assert result.decode_tps_max == result.decode_tps
        assert result.runs == 1
        assert result.approximate is True
        assert result.prefill_source == "ttft"
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
        _UsageHandler.timings = None
        _UsageHandler.cached_tokens = None
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_uses_upstream_timings_and_unique_prompts() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _UsageHandler.request_bodies = []
        _UsageHandler.timings = {
            "cache_n": 0,
            "prompt_n": 330,
            "prompt_ms": 816.236,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        model = ModelSpec("timing-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        result = measure(
            plan.services[0], f"http://127.0.0.1:{upstream.server_address[1]}",
            prefill_tokens=512, decode_tokens=17, runs=3,
        )
        assert result.prefill_source == "timings"
        assert result.prefill_tps == pytest.approx(404.29, rel=1e-3)
        assert result.decode_tps == pytest.approx(39.30, rel=1e-3)
        assert result.decode_tps_min == pytest.approx(39.30, rel=1e-3)
        assert result.decode_tps_max == pytest.approx(39.30, rel=1e-3)
        assert result.runs == 3
        assert result.approximate is False
        assert result.cached_prompt_tokens == 0
        prompts = [body["messages"][0]["content"] for body in _UsageHandler.request_bodies]
        assert len(prompts) == 3
        assert len(set(prompts)) == 3
        assert all(not prompt.startswith("benchmark filler text ") for prompt in prompts)
    finally:
        _UsageHandler.timings = None
        _UsageHandler.request_bodies = []
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_reports_cached_upstream_prompt() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        _UsageHandler.timings = {
            "cache_n": 24,
            "prompt_n": 314,
            "prompt_ms": 816.236,
            "predicted_n": 16,
            "predicted_ms": 381.695,
        }
        _UsageHandler.cached_tokens = 24
        model = ModelSpec("cached-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                          4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        result = measure(
            plan.services[0], f"http://127.0.0.1:{upstream.server_address[1]}",
            prefill_tokens=512, decode_tokens=17, runs=1,
        )
        assert result.prefill_source == "timings"
        assert result.cached_prompt_tokens == 24
    finally:
        _UsageHandler.timings = None
        _UsageHandler.cached_tokens = None
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_prefill_source_handles_partial_cache() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelSpec(
            "prefill-source-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
            4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
        )
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        cases = (
            ({"cache_n": 0, "prompt_n": 330, "prompt_ms": 816.236}, 0, "timings"),
            ({"cache_n": 24, "prompt_n": 314, "prompt_ms": 816.236}, 24, "timings"),
            ({"cache_n": 329, "prompt_n": 1, "prompt_ms": 3.0}, 329, "cached"),
            ({"cache_n": 5, "prompt_n": 10, "prompt_ms": 3.0}, 5, "cached"),
            (None, None, "ttft"),
        )
        for timings, cached, expected in cases:
            _UsageHandler.timings = timings
            _UsageHandler.cached_tokens = cached
            result = measure(
                plan.services[0], f"http://127.0.0.1:{upstream.server_address[1]}",
                prefill_tokens=512, decode_tokens=17, runs=1,
            )
            assert result.prefill_source == expected
    finally:
        _UsageHandler.timings = None
        _UsageHandler.cached_tokens = None
        upstream.shutdown()
        upstream.server_close()


def test_bench_runner_aggregates_decode_spread(monkeypatch) -> None:
    model = ModelSpec("spread-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    values = iter((12.0, 30.0, 18.0))

    def fake_measure_once(*_args, **_kwargs):
        decode_tps = next(values)
        return bench_runner.BenchResult(100.0, decode_tps, 0.2, False)

    monkeypatch.setattr(bench_runner, "_measure_once", fake_measure_once)
    result = measure(plan.services[0], "http://unused", runs=3)
    assert result.decode_tps == 18.0
    assert result.decode_tps_min == 12.0
    assert result.decode_tps_max == 30.0
    assert result.runs == 3


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


def test_tools_route_to_singular_tool_role_with_chat_fallback() -> None:
    model = ModelSpec("routing-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    plan = replace(
        plan,
        routing=replace(
            plan.routing,
            role_to_service={"chat": "chat-service", "tool": "tool-service"},
        ),
    )
    assert route({"tools": [{"type": "function"}]}, plan) == "tool-service"
    assert route({"tools": [{"type": "function"}]}, replace(
        plan,
        routing=replace(plan.routing, role_to_service={"chat": "chat-service"}),
    )) == "chat-service"


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


def _reserved_port() -> socket.socket:
    reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reserved.bind(("127.0.0.1", 0))
    return reserved


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
    # Bound-but-not-listening ports time out at CONNECT_TIMEOUT — keep the
    # retry path real but small enough for the suite.
    monkeypatch.setattr(gateway_module, "CONNECT_TIMEOUT", 0.2)
    with _reserved_port() as reserved:
        service = replace(service, port=reserved.getsockname()[1])
        isolated_plan = replace(plan, services=[service])
        with TestClient(create_app(isolated_plan)) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "nmesh-auto",
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert client.get("/metrics").json()["concurrency"][service.name]["in_flight"] == 0
    assert response.status_code == 504
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == 504
    assert len(calls) == 1
    assert calls[0] == (service.name, isolated_plan)


def test_gateway_retries_once_after_connect_timeout(monkeypatch) -> None:
    """ConnectTimeout is a TransportError sibling of ConnectError, not a
    subclass — a filtered or saturated port surfaces it instead of a clean
    refusal, and the revive+retry path must still run."""
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

    async def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(gateway_module, "ensure_running", fake_ensure)
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with TestClient(create_app(plan)) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "nmesh-auto",
            "messages": [{"role": "user", "content": "hello"}],
        })
    assert response.status_code == 504
    assert len(calls) == 1
    assert calls[0] == (service.name, plan)


def test_gateway_revive_failure_returns_503(monkeypatch) -> None:
    """A service whose restart budget is exhausted must surface the reason as
    503 — the request path may not resurrect it behind the circuit breaker."""
    model = ModelSpec("revive-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = plan.services[0]

    def fake_ensure(name: str, snapshot: Plan) -> None:
        raise RuntimeError(f"restart budget exhausted for {name}")

    monkeypatch.setattr(gateway_module, "ensure_running", fake_ensure)
    monkeypatch.setattr(
        gateway_module, "idle_services", lambda: {service.name}
    )
    client = TestClient(create_app(plan))
    response = client.post("/v1/chat/completions", json={
        "model": "nmesh-auto",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 503
    assert "restart budget" in response.json()["error"]["message"]


def test_gateway_injects_usage_for_vllm_and_ollama_streams(monkeypatch) -> None:
    """vllm/ollama support stream_options.include_usage (OpenAI spec), so the
    gateway injects it like llamacpp and records exact decode telemetry."""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        gateway_module, "_base_url",
        lambda service: f"http://127.0.0.1:{upstream.server_address[1]}",
    )
    samples = []
    monkeypatch.setattr(gateway_module, "record_telemetry", samples.append)
    try:
        model = ModelSpec("usage-backend-model", "test", 500_000_000, 24, 16, 2,
                          64, 1024, 4096, ["chat"], 80.0, "test",
                          {"hf_gguf": "test/repo"})
        plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
        for backend in ("vllm", "ollama"):
            _UsageHandler.request_body = {}
            service = replace(plan.services[0], backend=backend)
            client = TestClient(create_app(replace(plan, services=[service])))
            response = client.post("/v1/chat/completions", json={
                "model": "client-model", "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert response.status_code == 200
            assert _UsageHandler.request_body["stream_options"] == {
                "include_usage": True,
            }
            assert samples[-1].approximate is False
            assert samples[-1].completion_tokens == 17
    finally:
        _UsageHandler.request_body = {}
        upstream.shutdown()
        upstream.server_close()
