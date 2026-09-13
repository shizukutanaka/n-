from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from fastapi.testclient import TestClient

import nmesh.gateway as gateway_module
from nmesh.bench.embed import EMBED_HARNESS_VERSION, EmbedRecord, embed_key
from nmesh.bench.retrieval import (
    RETRIEVAL_HARNESS_VERSION,
    RetrievalChunkArm,
    RetrievalRecord,
    RetrievalRung,
    retrieval_digest,
    retrieval_key,
)
from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.planner import Plan, Policy, build_plan

from .test_planner import profile


class _EmbeddingHandler(BaseHTTPRequestHandler):
    request_bodies: ClassVar[list[dict[str, object]]] = []
    confirmation_tokens: ClassVar[int] = 3000

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.request_bodies.append(body)
        input_value = body.get("input")
        prompt_tokens = (
            self.__class__.confirmation_tokens
            if isinstance(input_value, list) and len(input_value) == 2
            else 2048
        )
        payload = {
            "data": [{"embedding": [0.1], "index": 0, "object": "embedding"}],
            "model": body["model"],
            "object": "list",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_upstream() -> ThreadingHTTPServer:
    _EmbeddingHandler.request_bodies = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    return upstream


def _plan(roles: list[str], port: int) -> Plan:
    model = ModelSpec(
        "embed" if roles == ["embed"] else "chat",
        "test",
        500_000_000,
        24,
        16,
        2,
        64,
        1024,
        4096,
        roles,
        80.0,
        "apache",
        {"hf_gguf": "org/model"},
    )
    plan = build_plan(profile(64), [model], Policy(roles=roles))
    service = replace(
        plan.services[0],
        backend="llamacpp",
        port=port,
        resident=True,
    )
    return replace(plan, services=[service], swap_group=[])


def _record() -> EmbedRecord:
    return EmbedRecord(
        "embed",
        "f16",
        "llamacpp",
        "cpu",
        0,
        8192,
        12288,
        2048,
        24576,
        2048,
        100.0,
        90.0,
        110.0,
        512,
        3,
        EMBED_HARNESS_VERSION,
        1.0,
    )


def _client(monkeypatch, upstream: ThreadingHTTPServer) -> TestClient:
    monkeypatch.setattr(gateway_module, "idle_services", list)
    monkeypatch.setattr(
        gateway_module,
        "load_embed_cache",
        lambda: {embed_key("embed", "f16", "llamacpp", "cpu", 0): _record()},
    )
    return TestClient(create_app(_plan(["embed"], upstream.server_address[1])))


class _AutoChunkHandler(BaseHTTPRequestHandler):
    request_bodies: ClassVar[list[dict[str, object]]] = []
    dimensions: ClassVar[int] = 2
    varying_dimensions: ClassVar[bool] = False
    prompt_tokens_override: ClassVar[int | None] = None

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.request_bodies.append(body)
        input_value = body.get("input")
        values = input_value if isinstance(input_value, list) else [input_value]
        data = []
        for index, value in enumerate(values):
            vector = [1.0, 0.0] if "needle" in str(value) else [0.0, 1.0]
            embedding = vector[:self.__class__.dimensions]
            if self.__class__.varying_dimensions and index:
                embedding = [embedding[0]]
            data.append({
                "embedding": embedding,
                "index": index,
                "object": "embedding",
            })
        tokens = sum(len(str(value).split()) for value in values)
        if self.__class__.prompt_tokens_override is not None:
            tokens = self.__class__.prompt_tokens_override
        payload = {
            "data": data,
            "model": body["model"],
            "object": "list",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_autochunk_upstream() -> ThreadingHTTPServer:
    _AutoChunkHandler.request_bodies = []
    _AutoChunkHandler.dimensions = 2
    _AutoChunkHandler.varying_dimensions = False
    _AutoChunkHandler.prompt_tokens_override = None
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _AutoChunkHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    return upstream


def _retrieval_record() -> RetrievalRecord:
    rungs = tuple(
        RetrievalRung(index, tokens, hits, 8, False)
        for index, (tokens, hits) in enumerate(
            ((100, 8), (400, 8), (800, 8), (1600, 8), (2400, 7), (3000, 1)),
            1,
        )
    )
    return RetrievalRecord(
        "embed", "f16", "llamacpp", "cpu", 0, rungs,
        retrieval_digest(), RETRIEVAL_HARNESS_VERSION, 2.0,
        RetrievalChunkArm(3000, 4, 8, 8, 8, 8, 8),
    )


def _autochunk_client(
    monkeypatch, upstream: ThreadingHTTPServer, record: RetrievalRecord | None,
) -> TestClient:
    monkeypatch.setenv("NMESH_EMBED_AUTOCHUNK", "1")
    monkeypatch.setattr(gateway_module, "idle_services", list)
    monkeypatch.setattr(
        gateway_module,
        "load_embed_cache",
        lambda: {embed_key("embed", "f16", "llamacpp", "cpu", 0): _record()},
    )
    monkeypatch.setattr(
        gateway_module,
        "load_retrieval_cache",
        lambda: (
            {retrieval_key(
                record.model_id, record.quant, record.backend, record.gpu_name,
                record.n_gpu_layers,
            ): record}
            if record is not None else {}
        ),
    )
    return TestClient(create_app(_plan(["embed"], upstream.server_address[1])))


def test_embedding_autochunk_disabled_preserves_body(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        monkeypatch.delenv("NMESH_EMBED_AUTOCHUNK", raising=False)
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            monkeypatch.delenv("NMESH_EMBED_AUTOCHUNK", raising=False)
            value = " ".join(["word"] * 10)
            response = client.post(
                "/v1/embeddings", json={"model": "nmesh-auto", "input": value},
            )
        assert response.status_code == 200
        assert _AutoChunkHandler.request_bodies[0]["input"] == value
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_pools_vectors_and_sets_headers(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            value = " ".join(["word"] * 7) + " needle"
            response = client.post(
                "/v1/embeddings", json={"model": "nmesh-auto", "input": value},
            )
        assert response.status_code == 200
        assert response.json()["data"][0]["embedding"] == [0.5, 0.5]
        assert response.headers["X-Nmesh-Embedding-Chunked"] == "2"
        assert response.headers["X-Nmesh-Embedding-Chunk-Words"] == "4"
        assert response.json()["usage"]["prompt_tokens"] == 8
        assert len(_AutoChunkHandler.request_bodies) == 1
        assert isinstance(_AutoChunkHandler.request_bodies[0]["input"], list)
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_without_evidence_preserves_body(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        with _autochunk_client(monkeypatch, upstream, None) as client:
            value = " ".join(["word"] * 10)
            response = client.post(
                "/v1/embeddings", json={"model": "nmesh-auto", "input": value},
            )
        assert response.status_code == 200
        assert _AutoChunkHandler.request_bodies[0]["input"] == value
        assert "X-Nmesh-Embedding-Chunked" not in response.headers
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_short_input_is_untouched(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": "short"},
            )
        assert response.status_code == 200
        assert _AutoChunkHandler.request_bodies[0]["input"] == "short"
        assert "X-Nmesh-Embedding-Chunked" not in response.headers
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_mixed_inputs_preserve_order(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": [
                    " ".join(["word"] * 7) + " needle", "short",
                ]},
            )
        assert response.status_code == 200
        assert [item["index"] for item in response.json()["data"]] == [0, 1]
        assert response.json()["data"][0]["embedding"] == [0.5, 0.5]
        assert response.json()["data"][1]["embedding"] == [0.0, 1.0]
        assert len(_AutoChunkHandler.request_bodies) == 2
        assert isinstance(_AutoChunkHandler.request_bodies[0]["input"], list)
        assert _AutoChunkHandler.request_bodies[1]["input"] == "short"
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_dimension_failure_falls_back(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        _AutoChunkHandler.varying_dimensions = True
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            value = " ".join(["word"] * 7) + " needle"
            response = client.post(
                "/v1/embeddings", json={"model": "nmesh-auto", "input": value},
            )
        assert response.status_code == 200
        assert response.json()["data"][0]["embedding"] == [1.0, 0.0]
        assert "X-Nmesh-Embedding-Chunked" not in response.headers
        assert len(_AutoChunkHandler.request_bodies) == 2
        assert _AutoChunkHandler.request_bodies[-1]["input"] == value
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_autochunk_mixed_inputs_skip_cap_guard(monkeypatch) -> None:
    upstream = _start_autochunk_upstream()
    try:
        _AutoChunkHandler.prompt_tokens_override = 3000
        with _autochunk_client(monkeypatch, upstream, _retrieval_record()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": [
                    " ".join(["word"] * 7) + " needle", "short",
                ]},
            )
        assert response.status_code == 200
        assert "X-Nmesh-Embedding-Truncation" not in response.headers
        assert response.headers["X-Nmesh-Embedding-Chunked"] == "3"
        assert response.headers["X-Nmesh-Embedding-Chunk-Words"] == "4"
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_truncation_returns_context_error(monkeypatch) -> None:
    upstream = _start_upstream()
    try:
        with _client(monkeypatch, upstream) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": " ".join(["word"] * 3000)},
            )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "context_length_exceeded"
        assert "2048" in error["message"]
        assert "3000" in error["message"]
        assert len(_EmbeddingHandler.request_bodies) == 2
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_saturation_within_tolerance_returns_original_body(
    monkeypatch,
) -> None:
    _EmbeddingHandler.confirmation_tokens = 2050
    upstream = _start_upstream()
    try:
        with _client(monkeypatch, upstream) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": " ".join(["word"] * 3000)},
            )
        assert response.status_code == 200
        assert response.json()["usage"]["prompt_tokens"] == 2048
        assert "X-Nmesh-Embedding-Truncation" not in response.headers
        assert len(_EmbeddingHandler.request_bodies) == 2
    finally:
        _EmbeddingHandler.confirmation_tokens = 3000
        upstream.shutdown()
        upstream.server_close()


def test_embedding_without_proven_cap_makes_one_upstream_request(monkeypatch) -> None:
    upstream = _start_upstream()
    try:
        monkeypatch.setattr(gateway_module, "idle_services", list)
        monkeypatch.setattr(gateway_module, "load_embed_cache", dict)
        with TestClient(
            create_app(_plan(["embed"], upstream.server_address[1]))
        ) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": " ".join(["word"] * 3000)},
            )
        assert response.status_code == 200
        assert len(_EmbeddingHandler.request_bodies) == 1
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_embedding_multi_input_is_marked_unverified(monkeypatch) -> None:
    upstream = _start_upstream()
    try:
        with _client(monkeypatch, upstream) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "nmesh-auto", "input": ["first", "second"]},
            )
        assert response.status_code == 200
        assert response.headers["X-Nmesh-Embedding-Truncation"] == "unverified"
        assert len(_EmbeddingHandler.request_bodies) == 1
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_non_embed_route_is_unchanged(monkeypatch) -> None:
    upstream = _start_upstream()
    try:
        monkeypatch.setattr(gateway_module, "idle_services", list)
        monkeypatch.setattr(
            gateway_module,
            "load_embed_cache",
            lambda: {embed_key("embed", "f16", "llamacpp", "cpu", 0): _record()},
        )
        with TestClient(
            create_app(_plan(["chat"], upstream.server_address[1]))
        ) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "nmesh-auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert response.status_code == 200
        assert "X-Nmesh-Embedding-Truncation" not in response.headers
        assert len(_EmbeddingHandler.request_bodies) == 1
    finally:
        upstream.shutdown()
        upstream.server_close()
