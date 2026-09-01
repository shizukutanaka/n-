from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from fastapi.testclient import TestClient

from nmesh.bench import measure
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.planner import Policy, build_plan

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
