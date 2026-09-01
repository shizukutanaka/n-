from __future__ import annotations

from types import SimpleNamespace

import pytest

from nmesh.runtime import acquisition


def _ollama_service() -> SimpleNamespace:
    return SimpleNamespace(
        backend="ollama",
        model_ref="qwen2.5:0.5b-instruct",
        model_id="probe-model",
        context=8192,
        name="chat",
    )


def test_ollama_acquisition_creates_context_model(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(argv, check):
        assert check
        calls.append(argv)

    monkeypatch.setattr(acquisition, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(acquisition.subprocess, "run", run)

    acquired = acquisition.acquire(_ollama_service())

    modelfile = tmp_path / "ollama" / "nmesh-probe-model-c8192.Modelfile"
    assert modelfile.read_text(encoding="utf-8") == (
        "FROM qwen2.5:0.5b-instruct\n"
        "PARAMETER num_ctx 8192\n"
    )
    assert calls == [
        ["ollama", "pull", "qwen2.5:0.5b-instruct"],
        ["ollama", "create", "nmesh-probe-model-c8192", "-f", str(modelfile)],
    ]
    assert acquired.model_ref == "nmesh-probe-model-c8192"
    assert acquired.path is None


def test_ollama_create_failure_returns_context_warning(tmp_path, monkeypatch) -> None:
    calls = 0

    def run(argv, check):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise acquisition.subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(acquisition, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(acquisition.subprocess, "run", run)

    acquired = acquisition.acquire(_ollama_service())

    assert acquired.model_ref == "qwen2.5:0.5b-instruct"
    assert acquired.path is None
    assert acquired.warning is not None
    assert "default context" in acquired.warning
    assert "8192" in acquired.warning


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
