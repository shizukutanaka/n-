import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import nmesh.planner.core as planner_core
from nmesh import cli
from nmesh.bench import benchmark_key
from nmesh.catalog import load_catalog
from nmesh.orchestrate.measure import RoleIdentity
from nmesh.planner import Policy, build_plan, save_plan
from nmesh.probe import GPUInfo, HardwareProfile, Tier
from nmesh.spec import (
    ALLOW,
    KIND_NGRAM,
    MIXED,
    NOT_FASTER,
    NOT_IDENTICAL,
    ClassEvidence,
    SpecConfig,
    SpecRecord,
    decide,
    engine_identity,
    load_cache,
    save,
)


def _record(*, identical: bool = True, speedup: float = 1.2) -> SpecRecord:
    target = RoleIdentity("target", "q4_k_m", "llamacpp", "artifact")
    return SpecRecord(
        target=target,
        spec=SpecConfig(kind=KIND_NGRAM, n_max=3),
        engine="llama.cpp",
        harness="spec-v1",
        repeats=2,
        classes=(
            ClassEvidence(
                name="copy",
                speedup=speedup,
                identical=identical,
                reference_tps=10.0,
                candidate_tps=speedup * 10.0,
                acceptance=0.8,
            ),
        ),
        at=1.0,
    )


def test_benchmark_key_preserves_none_and_separates_spec() -> None:
    base = benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0)
    assert base == "m|q4_k_m|llamacpp|cpu|0"
    assert benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0, spec="ngram") != base
    assert benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0, spec="draft") != base


def test_spec_record_save_load_round_trip(tmp_path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    loaded = load_cache(path)
    assert len(loaded) == 1
    restored = next(iter(loaded.values()))
    assert asdict(restored) == asdict(record)
    assert decide(restored) == (ALLOW, ALLOW)


def test_spec_decision_table_includes_mixed_regression() -> None:
    assert decide(_record()) == (ALLOW, ALLOW)
    assert decide(_record(identical=False)) == (NOT_IDENTICAL, NOT_IDENTICAL)
    assert decide(_record(speedup=1.0)) == (NOT_FASTER, NOT_FASTER)
    mixed = SpecRecord(
        target=_record().target,
        spec=_record().spec,
        engine="llama.cpp",
        harness="spec-v1",
        repeats=2,
        classes=(
            ClassEvidence("copy", 1.2, True, 10.0, 12.0, 1.0),
            ClassEvidence("prose", 0.9, True, 10.0, 9.0, 1.0),
        ),
        at=1.0,
    )
    assert decide(mixed) == (MIXED, MIXED)


def test_legacy_benchmark_entry_is_only_used_without_spec() -> None:
    model = load_catalog()[0]
    legacy = {(model.id, "q4_k_m", "llamacpp", "cpu", 0): 12.0}
    assert planner_core._bench_value(
        legacy, model, "q4_k_m", "llamacpp", "cpu", 0, "f16", "none"
    ) == 12.0
    assert planner_core._bench_value(
        legacy, model, "q4_k_m", "llamacpp", "cpu", 0, "f16", "ngram"
    ) is None


def test_engine_identity_round_trip_uses_the_shared_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr("nmesh.runtime.engine.active", lambda: None)
    profile = SimpleNamespace(available_backends={"llamacpp": "engine-build"})
    record = _record()
    save(record)
    assert engine_identity(profile) == "engine-build"
    assert next(iter(load_cache().values())) == record
    planned = _planned_spec(tmp_path, monkeypatch, decision="allow")
    assert "--spec-type" in planned.services[0].launch.argv


def test_spec_n_max_is_part_of_evidence_identity() -> None:
    from nmesh.spec import best_for

    record = _record()
    assert best_for(
        {record.spec.kind: record},
        record.target,
        SpecConfig(kind=KIND_NGRAM, n_max=3),
        record.engine,
    ) == record
    assert best_for(
        {record.spec.kind: record},
        record.target,
        SpecConfig(kind=KIND_NGRAM, n_max=5),
        record.engine,
    ) is None


def _llama_profile(
    flags: tuple[str, ...], *, gpu: bool = False
) -> HardwareProfile:
    gpus = (
        [GPUInfo(0, "Test GPU", "nvidia", 24 * 1024**3, 24 * 1024**3, (8, 0), False)]
        if gpu else []
    )
    return HardwareProfile(
        "windows", "Test CPU", 8, 8, 32 * 1024**3, 32 * 1024**3,
        100 * 1024**3, False, gpus,
        {"ollama": None, "llamacpp": "engine-build", "vllm": None, "mlx": None},
        Tier.T3_HIGH if gpu else Tier.T0_CPU,
        backend_flags={"llamacpp": flags},
        backend_gpu_devices={"llamacpp": ("0",)} if gpu else {},
    )


def _planned_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: str | None,
    flags: tuple[str, ...] = ("--spec-type",),
    policy: Policy | None = None,
    gpu: bool = False,
):
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr("nmesh.runtime.engine.active", lambda: None)
    model = next(
        item for item in load_catalog() if item.id == "qwen2.5-0.5b-instruct"
    )
    profile = _llama_profile(flags, gpu=gpu)
    base = build_plan(
        profile, [model], Policy(roles=["chat"], min_decode_tps=0, eval_evidence=False)
    )
    service = base.services[0]
    if decision is not None:
        evidence = _record()
        evidence = SpecRecord(
            target=RoleIdentity(service.model_id, service.quant, service.backend, ""),
            spec=evidence.spec,
            engine="engine-build",
            harness=evidence.harness,
            repeats=evidence.repeats,
            classes=(
                ClassEvidence(
                    "copy",
                    {"not_identical": 1.2, "not_faster": 1.0, "mixed": 1.2}.get(
                        decision, 1.2
                    ),
                    decision != "not_identical",
                    10.0,
                    12.0,
                    1.0,
                ),
                *(
                    (ClassEvidence("prose", 0.9, True, 10.0, 9.0, 1.0),)
                    if decision == "mixed" else ()
                ),
            ),
            at=1.0,
        )
        save(evidence)
    result = build_plan(
        profile,
        [model],
        policy
        or Policy(
            roles=["chat"],
            min_decode_tps=0,
            eval_evidence=False,
            spec=KIND_NGRAM,
        ),
    )
    return result


@pytest.mark.parametrize(
    ("decision", "reason"),
    [(None, "no_evidence"), ("not_identical", "not_identical"),
     ("not_faster", "not_faster"), ("mixed", "mixed")],
)
def test_spec_flags_require_allow_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str | None,
    reason: str,
) -> None:
    result = _planned_spec(tmp_path, monkeypatch, decision=decision)
    service = result.services[0]
    assert "--spec-type" not in service.launch.argv
    assert any(reason in warning for warning in result.warnings)


def test_spec_flags_are_emitted_for_allow_and_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = _planned_spec(tmp_path, monkeypatch, decision="allow")
    assert "--spec-type" in allowed.services[0].launch.argv
    overridden = _planned_spec(
        tmp_path,
        monkeypatch,
        decision=None,
        policy=Policy(
            roles=["chat"], min_decode_tps=0, eval_evidence=False,
            spec=KIND_NGRAM, ignore_spec_evidence=True,
        ),
    )
    assert "--spec-type" in overridden.services[0].launch.argv
    assert any("bypassed" in warning for warning in overridden.warnings)


def test_spec_capabilities_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _planned_spec(
        tmp_path, monkeypatch, decision="allow", flags=("--parallel",)
    )
    assert "--spec-type" not in result.services[0].launch.argv
    assert any("unsupported" in warning for warning in result.warnings)


def test_draft_capabilities_require_all_draft_flags() -> None:
    assert not planner_core._honors_spec(
        "llamacpp", ("--spec-type", "--spec-draft-n-max"), "draft"
    )


def test_default_plan_serialization_omits_spec_fields(
    tmp_path: Path,
) -> None:
    result = build_plan(
        _llama_profile(("--parallel",)),
        [next(item for item in load_catalog() if item.id == "qwen2.5-0.5b-instruct")],
        Policy(roles=["chat"], min_decode_tps=0, eval_evidence=False),
    )
    path = tmp_path / "plan.json"
    save_plan(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "spec" not in payload["policy"]
    assert "spec_draft" not in payload["policy"]
    assert "spec_n_max" not in payload["policy"]
    assert "ignore_spec_evidence" not in payload["policy"]
    assert "spec" not in payload["services"][0]


def test_missing_and_gpu_draft_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = _planned_spec(
        tmp_path,
        monkeypatch,
        decision=None,
        flags=("--spec-type", "--spec-draft-model", "--spec-draft-n-max"),
        policy=Policy(
            roles=["chat"], min_decode_tps=0, eval_evidence=False,
            spec="draft", spec_draft="missing-draft",
        ),
    )
    assert any("not found" in warning for warning in missing.warnings)
    draft = tmp_path / "models" / "draft.gguf"
    draft.parent.mkdir()
    draft.write_bytes(b"draft")
    monkeypatch.setattr("nmesh.planner.core._spec_draft_path", lambda _value: draft)
    gpu = _planned_spec(
        tmp_path,
        monkeypatch,
        decision=None,
        flags=("--spec-type", "--spec-draft-model", "--spec-draft-n-max", "-ngl"),
        policy=Policy(
            roles=["chat"], min_decode_tps=0, eval_evidence=False,
            spec="draft", spec_draft="draft",
        ),
        gpu=True,
    )
    assert any("GPU" in warning or "GPU" in warning.upper() for warning in gpu.warnings)


def test_draft_path_requires_exact_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    exact = models / "draft-model.gguf"
    exact.write_bytes(b"draft")
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    assert planner_core._spec_draft_path("draft") is None
    assert planner_core._spec_draft_path("DRAFT-MODEL") == exact.resolve()


def test_spec_cli_validates_draft_and_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(Path.cwd() / "missing-spec-home"))
    assert cli.main(["spec", "measure", "--kind", "draft"]) == 2
    assert "--draft" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["spec", "measure", "--kind", "unknown"])
