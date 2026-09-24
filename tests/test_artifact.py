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


def _nested_array_gguf(key: str, depth: int) -> bytes:
    return b"".join((
        b"GGUF",
        struct.pack("<IQQ", 3, 0, 1),
        _string(key),
        struct.pack("<I", 9),
        # GGUF forbids arrays of arrays: each item declares its own
        # item_kind and count, so a hostile header recurses once per level.
        b"".join(struct.pack("<IQ", 9, 1) for _ in range(depth)),
    ))


def test_gguf_info_rejects_nested_arrays_instead_of_recursing(tmp_path: Path) -> None:
    path = tmp_path / "nested.gguf"
    # Value path: a tracked KV key whose array-of-arrays would recurse past
    # the interpreter limit before the file could possibly end.
    path.write_bytes(_nested_array_gguf("llama.block_count", 1200))
    assert artifact.gguf_info(path) is None
    # Skip path: the same shape behind an untracked key.
    path.write_bytes(_nested_array_gguf("untracked.key", 1200))
    assert artifact.gguf_info(path) is None


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


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_u32_array(key: str, values: tuple[int, ...]) -> bytes:
    return (
        _string(key)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 4, len(values))
        + b"".join(struct.pack("<I", item) for item in values)
    )


def _kv_bool_array(key: str, values: tuple[bool, ...]) -> bytes:
    return (
        _string(key)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 7, len(values))
        + b"".join(b"\x01" if item else b"\x00" for item in values)
    )


def _attention_gguf(*, pattern_array: bool) -> bytes:
    return b"".join((
        b"GGUF",
        struct.pack("<IQQ", 3, 0, 8),
        _kv_string("general.architecture", "gptoss"),
        _kv_u32("gptoss.block_count", 24),
        _kv_u32_array("gptoss.attention.head_count_kv", (8,) * 24),
        _kv_u32("gptoss.attention.key_length", 128),
        _kv_u32("gptoss.embedding_length", 2880),
        _kv_u32("gptoss.attention.sliding_window", 128),
        (
            _kv_bool_array(
                "gptoss.attention.sliding_window_pattern",
                tuple(index % 2 == 0 for index in range(24)),
            )
            if pattern_array
            else _kv_u32("gptoss.attention.sliding_window_pattern", 2)
        ),
        _string("unrelated.key"),
        struct.pack("<I", 8),
        _string("skipped"),
    ))


def test_gguf_info_collects_attention_layout(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_attention_gguf(pattern_array=True))
    info = artifact.gguf_info(path)
    assert info is not None
    assert info.block_count == 24
    assert info.head_count_kv == 8
    assert info.key_length == 128
    assert info.embedding_length == 2880
    assert info.sliding_window == 128
    # Per-layer bool pattern: every other layer slides, 12 of 24.
    assert info.swa_layers == 12


def test_gguf_info_scalar_pattern_counts_sliding_layers(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_attention_gguf(pattern_array=False))
    info = artifact.gguf_info(path)
    assert info is not None
    # Scalar pattern 2: every second layer is full attention.
    assert info.swa_layers == 12


def test_gguf_info_without_attention_keys_defaults_to_zero(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_synthetic_gguf())
    info = artifact.gguf_info(path)
    assert info is not None
    assert info.block_count == 0
    assert info.swa_layers == 0


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
