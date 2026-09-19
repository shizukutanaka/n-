from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
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


def test_parse_label_canonicalizes_full_vocabulary() -> None:
    cases = {
        "gemma-2-9b-it-Q3_K_L-Q8.gguf": "q3_k_l+q8",
        "gemma-2-9b-it-Q4_K_M-fp16.gguf": "q4_k_m+f16",
        "gemma-2-9b-it-Q8_0-f16.gguf": "q8_0+f16",
        "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf": "q4_0_4_4",
        "model-IQ4_XS.gguf": "iq4_xs",
        "model-Q4_K_S.gguf": "q4_k_s",
        "model-Q3_K_XL.gguf": "q3_k_xl",
        "model-Q8_0_L.gguf": "q8_0_l",
        "model-Q5_0.gguf": "q5_0",
        "model-f32.gguf": "f32",
        "bge-m3-Q3_K.gguf": "q3_k",
    }
    for filename, expected in cases.items():
        assert acquisition.parse_label(filename) == expected
    assert acquisition.parse_label("model-without-quant.gguf") is None


def test_gguf_files_reads_names_and_sizes_from_one_metadata_call(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class Api:
        def model_info(self, repo_id, files_metadata):
            calls.append((repo_id, files_metadata))
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=123),
                    SimpleNamespace(rfilename="README.md", size=456),
                    SimpleNamespace(
                        rfilename="model-Q4_K_M-00001-of-00002.gguf",
                        lfs=SimpleNamespace(size=456),
                    ),
                ]
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert acquisition._gguf_files("repo") == {
        "model-Q4_K_M.gguf": 123,
        "model-Q4_K_M-00001-of-00002.gguf": 456,
    }
    assert calls == [("repo", True)]


def test_resolve_exact_quant(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: {
            "model-q4_k_m.gguf": 10,
            "model-q4_k_s.gguf": 8,
        },
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_m",
        ["model-q4_k_m.gguf"],
        10,
    )


def test_resolve_quant_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition, "_gguf_files", lambda _repo: {"model-FP16.GGUF": 10}
    )
    assert acquisition._resolve_gguf("repo", "f16") == ("f16", ["model-FP16.GGUF"], 10)


def test_resolve_accepts_newly_supported_lower_quant(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition, "_gguf_files", lambda _repo: {"model-q4_k_s.gguf": 10}
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_s",
        ["model-q4_k_s.gguf"],
        10,
    )


def test_resolve_downgrades_to_highest_safe_quant(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files", lambda _repo: {
            "model-fp16.gguf": 10,
            "model-q4_0.gguf": 8,
            "model-q3_k_m.gguf": 6,
        },
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_0",
        ["model-q4_0.gguf"],
        8,
    )


def test_resolve_accepts_mxfp4_under_q4_k_m_plan(monkeypatch) -> None:
    # gpt-oss-* GGUFs ship MXFP4 only; a default q4_k_m plan must resolve them.
    monkeypatch.setattr(
        acquisition, "_gguf_files", lambda _repo: {"gpt-oss-20b-MXFP4.gguf": 10}
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "mxfp4",
        ["gpt-oss-20b-MXFP4.gguf"],
        10,
    )


def test_resolve_reports_published_quants_when_none_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition, "_gguf_files", lambda _repo: {"model-fp16.gguf": 10}
    )
    with pytest.raises(RuntimeError, match=r"repo.*published labels: f16"):
        acquisition._resolve_gguf("repo", "q4_k_m")


def test_resolve_returns_complete_split_set(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: {
            "model-q4_k_m-00002-of-00002.gguf": 6,
            "model-q4_k_m-00001-of-00002.gguf": 4,
        },
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_m",
        [
            "model-q4_k_m-00001-of-00002.gguf",
            "model-q4_k_m-00002-of-00002.gguf",
        ],
        10,
    )


def test_resolve_skips_incomplete_split_set(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "_gguf_files",
        lambda _repo: {"model-q4_k_m-00001-of-00002.gguf": 4},
    )
    with pytest.raises(RuntimeError, match="published labels: q4_k_m"):
        acquisition._resolve_gguf("repo", "q4_k_m")


def test_resolve_real_gemma_labels_and_prefers_plain_artifacts(monkeypatch) -> None:
    files = {
        "gemma-2-9b-it-Q8_0.gguf": 8,
        "gemma-2-9b-it-Q3_K_L-Q8.gguf": 3,
        "gemma-2-9b-it-Q6_K.gguf": 6,
        "gemma-2-9b-it-Q6_K-Q8.gguf": 5,
        "gemma-2-9b-it-Q4_K_M.gguf": 4,
        "gemma-2-9b-it-Q4_K_M-fp16.gguf": 7,
    }
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: files)

    assert acquisition._resolve_gguf("repo", "q8_0") == (
        "q8_0", ["gemma-2-9b-it-Q8_0.gguf"], 8,
    )
    assert acquisition._resolve_gguf("repo", "q6_k") == (
        "q6_k", ["gemma-2-9b-it-Q6_K.gguf"], 6,
    )
    assert acquisition._resolve_gguf("repo", "q4_k_m") == (
        "q4_k_m", ["gemma-2-9b-it-Q4_K_M.gguf"], 4,
    )
    assert acquisition._resolve_gguf("repo", "f16") == (
        "q8_0", ["gemma-2-9b-it-Q8_0.gguf"], 8,
    )


def test_resolve_excludes_cpu_repack_and_uses_safe_fallback(monkeypatch) -> None:
    files = {
        "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf": 44,
        "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf": 33,
    }
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: files)
    assert acquisition._resolve_gguf("repo", "q4_0") == (
        "q3_k_m", ["Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf"], 33,
    )


def test_resolve_uses_iq_and_k_s_vocabulary(monkeypatch) -> None:
    files = {
        "gemma-2-2b-it-IQ4_XS.gguf": 14,
        "gemma-2-2b-it-Q4_K_S.gguf": 13,
        "gemma-2-2b-it-Q3_K_L.gguf": 12,
        "gemma-2-2b-it-IQ3_M.gguf": 11,
        "gemma-2-2b-it-Q5_K_S.gguf": 15,
    }
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: files)
    assert acquisition._resolve_gguf("repo", "q3_k_m") == (
        "iq3_m", ["gemma-2-2b-it-IQ3_M.gguf"], 11,
    )


def test_resolve_does_not_rank_bare_k_label(monkeypatch) -> None:
    monkeypatch.setattr(
        acquisition, "_gguf_files", lambda _repo: {"bge-m3-Q3_K.gguf": 11}
    )
    with pytest.raises(RuntimeError, match="published labels: q3_k"):
        acquisition._resolve_gguf("repo", "q3_k_m")


def test_resolve_accepts_nominal_non_planner_label(monkeypatch) -> None:
    files = {
        "model-IQ3_M.gguf": 11,
        "model-IQ2_M.gguf": 9,
    }
    monkeypatch.setattr(acquisition, "_gguf_files", lambda _repo: files)
    assert acquisition._resolve_gguf("repo", "iq3_m") == (
        "iq3_m", ["model-IQ3_M.gguf"], 11,
    )


def _llamacpp_service(tmp_path: Path, *, quant: str = "q4_k_m") -> SimpleNamespace:
    return SimpleNamespace(
        backend="llamacpp",
        model_ref=str(tmp_path / "planned.gguf"),
        download_repo="repo",
        quant=quant,
        name="chat",
        memory=SimpleNamespace(weight_bytes=100),
    )


def test_mixed_artifact_warning_and_size_mismatch(tmp_path, monkeypatch) -> None:
    service = _llamacpp_service(tmp_path)
    monkeypatch.setattr(
        acquisition,
        "_resolve_gguf",
        lambda _repo, _quant: ("q3_k_l+q8", ["model-Q3_K_L-Q8.gguf"], 150),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: str(tmp_path / "model-Q3_K_L-Q8.gguf"),
    )
    acquired = acquisition.acquire(service)
    assert acquired.quant == "q3_k_l+q8"
    assert acquired.artifact_bytes == 150
    assert acquired.warning is not None
    assert "mixed-precision" in acquired.warning
    assert "150" in acquired.warning and "100" in acquired.warning


def test_existing_local_gguf_reports_parsed_identity_and_size(tmp_path) -> None:
    target = tmp_path / "model-Q3_K_L-Q8.gguf"
    target.write_bytes(b"artifact")
    service = _llamacpp_service(tmp_path)
    service.model_ref = str(target)

    acquired = acquisition.acquire(service)

    assert acquired.path == target
    assert acquired.quant == "q3_k_l+q8"
    assert acquired.substituted is True
    assert acquired.artifact_bytes == len(b"artifact")


def test_size_mismatch_warning_is_absent_at_parity(tmp_path, monkeypatch) -> None:
    service = _llamacpp_service(tmp_path)
    monkeypatch.setattr(
        acquisition,
        "_resolve_gguf",
        lambda _repo, _quant: ("q4_k_m", ["model-Q4_K_M.gguf"], 100),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: str(tmp_path / "model-Q4_K_M.gguf"),
    )
    acquired = acquisition.acquire(service)
    assert acquired.warning is None
