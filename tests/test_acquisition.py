from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from nmesh import artifacts
from nmesh.runtime import acquisition


@pytest.fixture(autouse=True)
def _isolated_artifact_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "CACHE_PATH", tmp_path / "artifacts.json")


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

    def run(argv, check, env=None):
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

    def run(argv, check, env=None):
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


def test_interrupted_split_gguf_resumes_missing_parts(tmp_path, monkeypatch) -> None:
    """part 1 on disk + part 2 still incomplete must not count as present —
    acquire should fall through to the download path instead of adopting the
    partial artifact."""
    target = tmp_path / "model-00001-of-00002.gguf"
    target.write_bytes(b"part1")
    service = _llamacpp_service(tmp_path)
    service.model_ref = str(target)
    service.download_repo = "org/repo"

    calls: list[str] = []

    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        calls.append(filename)
        path = tmp_path / filename
        if not path.exists():
            path.write_bytes(b"part")
        return str(path)

    monkeypatch.setattr(
        acquisition,
        "_resolve_gguf",
        lambda _repo, _quant: (
            "q4_k_m",
            ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"],
            200,
        ),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    acquired = acquisition.acquire(service)

    assert calls == [
        "model-00001-of-00002.gguf", "model-00002-of-00002.gguf"
    ]
    assert acquired.path == tmp_path / "model-00001-of-00002.gguf"


def test_complete_split_gguf_skips_download(tmp_path, monkeypatch) -> None:
    target = tmp_path / "model-00001-of-00002.gguf"
    target.write_bytes(b"part1")
    (tmp_path / "model-00002-of-00002.gguf").write_bytes(b"part2")
    service = _llamacpp_service(tmp_path)
    service.model_ref = str(target)

    acquired = acquisition.acquire(service)

    assert acquired.path == target


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


def test_truncated_cached_gguf_is_reacquired(tmp_path, monkeypatch) -> None:
    """A cached file whose bytes differ from the recorded download size is
    corrupt — acquire must re-download instead of launching against it."""
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(b"short")
    service = _llamacpp_service(tmp_path)
    service.model_ref = str(target)
    service.download_repo = "org/repo"

    from nmesh.artifacts import artifact_key

    monkeypatch.setattr(
        acquisition,
        "load_cache",
        lambda *a, **k: {artifact_key("org/repo", "q4_k_m"): 2000},
    )
    calls: list[str] = []

    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        calls.append(filename)
        path = tmp_path / filename
        path.write_bytes(b"x" * 2000)
        return str(path)

    monkeypatch.setattr(
        acquisition,
        "_resolve_gguf",
        lambda _repo, _quant: ("q4_k_m", ["model-Q4_K_M.gguf"], 2000),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    acquired = acquisition.acquire(service)

    assert calls == ["model-Q4_K_M.gguf"]
    assert acquired.path == target
    assert acquired.warning is not None and "corrupt" in acquired.warning


def test_unrecorded_cached_gguf_still_adopted(tmp_path, monkeypatch) -> None:
    """Hand-placed files have no recorded size — adopt as before."""
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(b"artifact")
    service = _llamacpp_service(tmp_path)
    service.model_ref = str(target)

    monkeypatch.setattr(acquisition, "load_cache", lambda *a, **k: {})

    acquired = acquisition.acquire(service)

    assert acquired.path == target


def test_local_only_adopts_matching_gguf(tmp_path, monkeypatch) -> None:
    """Planned names ({id}-{quant}.gguf) differ from downloaded filenames,
    so a no-download acquire must resolve the real on-disk artifact by
    model id + quant instead of pointing llama-server at a missing file."""
    service = _llamacpp_service(tmp_path)
    service.model_id = "qwen3-0.6b"
    real = tmp_path / "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
    real.write_bytes(b"artifact")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("network touched"),
    )

    acquired = acquisition.acquire(service, local_only=True)

    assert acquired.path == real
    assert acquired.quant == "q4_k_m"
    assert acquired.substituted is False


def test_local_only_falls_back_to_nearest_quant(tmp_path) -> None:
    """The exact quant may be absent while a lower one is on disk — adopt
    it as a substitution, mirroring _resolve_gguf's ranking."""
    service = _llamacpp_service(tmp_path, quant="q8_0")
    service.model_id = "qwen3-0.6b"
    real = tmp_path / "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
    real.write_bytes(b"artifact")

    acquired = acquisition.acquire(service, local_only=True)

    assert acquired.path == real
    assert acquired.quant == "q4_k_m"
    assert acquired.substituted is True


def test_local_only_without_match_raises(tmp_path) -> None:
    """Nothing usable on disk → fail loudly; silently launching the planned
    name would crash the engine."""
    service = _llamacpp_service(tmp_path)
    service.model_id = "qwen3-0.6b"

    with pytest.raises(RuntimeError, match="downloads are disabled"):
        acquisition.acquire(service, local_only=True)


def test_local_only_ollama_skips_pull(tmp_path, monkeypatch) -> None:
    """pull is the network step; create is local and must still run so the
    derived context model exists."""
    calls: list[list[str]] = []

    def run(argv, check, env=None):
        calls.append(argv)

    monkeypatch.setattr(acquisition, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(acquisition.subprocess, "run", run)

    acquisition.acquire(_ollama_service(), local_only=True)

    assert [argv[1] for argv in calls] == ["create"]


def test_ollama_acquisition_pins_loopback_host(tmp_path, monkeypatch) -> None:
    """A user-set OLLAMA_HOST would send pull/create to a daemon the plan
    does not manage; the CLI must be pinned to the managed loopback daemon."""
    envs: list[dict] = []

    def run(argv, check, env=None):
        envs.append(env)

    monkeypatch.setenv("OLLAMA_HOST", "http://remote-host:11434")
    monkeypatch.setattr(acquisition, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(acquisition.subprocess, "run", run)

    acquisition.acquire(_ollama_service())

    assert len(envs) == 2
    for env in envs:
        assert env is not None
        assert env["OLLAMA_HOST"] == "127.0.0.1:11434"


def test_enable_hf_transfer_sets_env_and_patches_loaded_constants(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(
        acquisition.importlib.util, "find_spec", lambda name: object()
    )
    constants = SimpleNamespace(HF_HUB_ENABLE_HF_TRANSFER=False)
    monkeypatch.setitem(
        acquisition.sys.modules, "huggingface_hub.constants", constants
    )
    acquisition._enable_hf_transfer()
    assert acquisition.os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert constants.HF_HUB_ENABLE_HF_TRANSFER is True


def test_enable_hf_transfer_respects_user_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    monkeypatch.setattr(
        acquisition.importlib.util, "find_spec", lambda name: object()
    )
    acquisition._enable_hf_transfer()
    assert acquisition.os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_enable_hf_transfer_noop_without_package(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(
        acquisition.importlib.util, "find_spec", lambda name: None
    )
    acquisition._enable_hf_transfer()
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in acquisition.os.environ
