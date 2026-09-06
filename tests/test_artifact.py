from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import httpx

from nmesh import artifact


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _synthetic_gguf() -> bytes:
    return b"".join((
        b"GGUF",
        struct.pack("<IQQ", 3, 1, 2),
        _string("general.architecture"),
        struct.pack("<I", 8),
        _string("qwen2"),
        _string("test.array"),
        struct.pack("<I", 9),
        struct.pack("<IQ", 4, 2),
        struct.pack("<II", 7, 8),
        _string("token_embd.weight"),
        struct.pack("<I", 2),
        struct.pack("<QQ", 2, 3),
        struct.pack("<IQ", 1, 0),
    ))


def test_gguf_fingerprint_reads_a_stable_header_only(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    payload = _synthetic_gguf() + b"tensor data that is not hashed"
    path.write_bytes(payload)
    expected = (
        f"gguf:1:{len(payload)}:"
        f"{hashlib.sha256(_synthetic_gguf()).hexdigest()[:16]}"
    )
    assert artifact.gguf_fingerprint(path) == expected
    path.write_bytes(payload[:10])
    assert artifact.gguf_fingerprint(path) is None
    path.write_bytes(b"not a GGUF file")
    assert artifact.gguf_fingerprint(path) is None


class _TagsResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> object:
        return self.payload


def test_ollama_fingerprint_matches_latest_and_handles_failures(monkeypatch) -> None:
    digest = "sha256:123456789012345678901234567890"
    monkeypatch.setattr(
        artifact.httpx,
        "get",
        lambda url, timeout: _TagsResponse({
            "models": [{"name": "qwen:latest", "digest": digest, "size": 42}],
        }),
    )
    assert artifact.ollama_fingerprint("qwen") == f"ollama:{digest[:19]}:42"
    monkeypatch.setattr(
        artifact.httpx,
        "get",
        lambda url, timeout: _TagsResponse({"models": []}),
    )
    assert artifact.ollama_fingerprint("missing") is None
    monkeypatch.setattr(
        artifact.httpx,
        "get",
        lambda url, timeout: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )
    assert artifact.ollama_fingerprint("qwen") is None


def test_service_fingerprint_dispatches_only_verified_backends(
    monkeypatch, tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        artifact,
        "gguf_fingerprint",
        lambda path: calls.append(("gguf", path)) or "gguf:fingerprint",
    )
    monkeypatch.setattr(
        artifact,
        "ollama_fingerprint",
        lambda model_ref: calls.append(("ollama", model_ref)) or "ollama:fingerprint",
    )
    assert artifact.service_fingerprint("llamacpp", str(tmp_path / "model.gguf")) == (
        "gguf:fingerprint"
    )
    assert artifact.service_fingerprint("ollama", "qwen") == "ollama:fingerprint"
    assert artifact.service_fingerprint("vllm", "qwen") is None
    assert calls[0][0] == "gguf"
    assert calls[1] == ("ollama", "qwen")
