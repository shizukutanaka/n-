from __future__ import annotations

import pytest

from nmesh.runtime import acquisition


def test_resolve_exact_quant(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: ["model-q4_k_m.gguf", "model-q4_k_s.gguf"],
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_m",
        ["model-q4_k_m.gguf"],
    )


def test_resolve_quant_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: ["model-FP16.GGUF"])
    assert acquisition._resolve_gguf("repo", "f16") == ("f16", ["model-FP16.GGUF"])


def test_resolve_does_not_match_similar_quant(monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: ["model-q4_k_s.gguf"])
    with pytest.raises(RuntimeError, match="published quants: none"):
        acquisition._resolve_gguf("repo", "q4_k_m")


def test_resolve_downgrades_to_highest_safe_quant(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: ["model-fp16.gguf", "model-q4_0.gguf", "model-q3_k_m.gguf"],
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_0",
        ["model-q4_0.gguf"],
    )


def test_resolve_reports_published_quants_when_none_fit(monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: ["model-fp16.gguf"])
    with pytest.raises(RuntimeError, match=r"repo.*f16"):
        acquisition._resolve_gguf("repo", "q4_k_m")


def test_resolve_returns_complete_split_set(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: [
            "model-q4_k_m-00002-of-00002.gguf",
            "model-q4_k_m-00001-of-00002.gguf",
        ],
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_m",
        [
            "model-q4_k_m-00001-of-00002.gguf",
            "model-q4_k_m-00002-of-00002.gguf",
        ],
    )
