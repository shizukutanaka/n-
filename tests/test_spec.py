import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import nmesh.planner.core as planner_core
import nmesh.spec.measure as spec_measure
from nmesh import cli
from nmesh.bench import EPOCH_MIN_RATIO, benchmark_key
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
    SPEC_HARNESS_VERSION,
    STALE,
    UNSTABLE,
    ArmRun,
    ClassEvidence,
    ClassResult,
    ControlEvidence,
    SpecConfig,
    SpecRecord,
    Workload,
    control,
    decide,
    demote_stale,
    engine_identity,
    from_arms,
    load_cache,
    save,
)


def _record(
    *,
    identical: bool = True,
    speedup: float = 1.2,
    epoch: str = "unknown",
    reference_id: str = "",
    reference_tps: float = 0.0,
) -> SpecRecord:
    target = RoleIdentity("target", "q4_k_m", "llamacpp", "artifact")
    return SpecRecord(
        target=target,
        spec=SpecConfig(kind=KIND_NGRAM, n_max=3),
        engine="llama.cpp",
        harness=SPEC_HARNESS_VERSION,
        repeats=3,
        classes=(
            ClassEvidence(
                name="copy",
                speedup=speedup,
                identical=identical,
                reference_tps=10.0,
                candidate_tps=speedup * 10.0,
                acceptance=0.8,
                reference_spread=0.12,
                candidate_spread=0.24,
            ),
        ),
        control=(ControlEvidence("copy", 1.0, True),),
        epoch=epoch,
        reference_id=reference_id,
        reference_tps=reference_tps,
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
    assert restored.classes[0].reference_spread == 0.12
    assert restored.classes[0].candidate_spread == 0.24
    assert decide(restored) == (ALLOW, ALLOW)


def test_missing_control_is_unstable(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry.pop("control")
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = next(iter(load_cache(path).values()))
    assert restored.control == ()
    assert decide(restored) == (UNSTABLE, UNSTABLE)


def test_malformed_control_is_rejected(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["control"] = [{"name": "copy", "ratio": "bad"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path) == {}


def test_legacy_two_repeat_record_is_rejected(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["repeats"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path) == {}


def test_legacy_record_without_spreads_defaults_to_zero(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["classes"][0].pop("reference_spread")
    entry["classes"][0].pop("candidate_spread")
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = next(iter(load_cache(path).values()))
    assert restored.classes[0].reference_spread == 0.0
    assert restored.classes[0].candidate_spread == 0.0


def test_legacy_record_without_epoch_fields_defaults_and_allows(
    tmp_path: Path,
) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    for field in ("epoch", "reference_id", "reference_tps"):
        entry.pop(field)
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = next(iter(load_cache(path).values()))
    assert restored.epoch == "unknown"
    assert restored.reference_id == ""
    assert restored.reference_tps == 0.0
    assert decide(restored) == (ALLOW, ALLOW)


@pytest.mark.parametrize(
    ("epoch", "speedup", "identical", "control", "expected"),
    [
        ("degraded", 1.2, True, True, STALE),
        ("degraded", 1.2, False, True, NOT_IDENTICAL),
        ("degraded", 1.2, True, False, UNSTABLE),
        ("degraded", 1.0, True, True, NOT_FASTER),
    ],
)
def test_degraded_epoch_preserves_stronger_rejections(
    epoch: str,
    speedup: float,
    identical: bool,
    control: bool,
    expected: str,
) -> None:
    record = _record(
        epoch=epoch,
        speedup=speedup,
        identical=identical,
    )
    if not control:
        record = replace(
            record, control=(ControlEvidence("copy", 0.5, True),)
        )
    assert decide(record) == (expected, expected)


def test_degraded_epoch_mixed_and_healthy_allow() -> None:
    mixed = SpecRecord(
        target=_record().target,
        spec=_record().spec,
        engine="llama.cpp",
        harness=SPEC_HARNESS_VERSION,
        repeats=3,
        classes=(
            ClassEvidence("copy", 1.2, True, 10.0, 12.0, 1.0),
            ClassEvidence("prose", 0.9, True, 10.0, 9.0, 1.0),
        ),
        control=(ControlEvidence("copy", 1.0, True),),
        epoch="degraded",
    )
    assert decide(mixed) == (MIXED, MIXED)
    assert decide(_record(epoch="healthy")) == (ALLOW, ALLOW)


@pytest.mark.parametrize(
    "field_value",
    ["invalid", 1, None],
)
def test_invalid_epoch_is_rejected(tmp_path: Path, field_value: object) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["epoch"] = field_value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path) == {}


@pytest.mark.parametrize("value", [-1, float("inf"), True])
def test_invalid_reference_tps_is_rejected(tmp_path: Path, value: object) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["reference_tps"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path) == {}


def test_invalid_reference_id_is_rejected(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["results"][next(iter(payload["results"]))]
    entry["reference_id"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path) == {}


def test_spec_demote_stale_uses_epoch_constant() -> None:
    records = {
        "old": _record(reference_id="ref", reference_tps=20.0),
        "noise": _record(reference_id="ref", reference_tps=45.0),
        "legacy": _record(reference_id="", reference_tps=20.0),
        "other": _record(reference_id="other", reference_tps=20.0),
        "zero": _record(reference_id="ref", reference_tps=0.0),
    }
    demoted = demote_stale(
        records,
        "ref",
        20.0 / EPOCH_MIN_RATIO,
    )
    assert demoted == ("old",)
    assert records["old"].epoch == "degraded"
    assert records["noise"].epoch == "unknown"
    assert records["legacy"].epoch == "unknown"
    assert records["other"].epoch == "unknown"
    assert records["zero"].epoch == "unknown"


def test_control_failure_is_unstable() -> None:
    assert decide(
        replace(_record(), control=(ControlEvidence("copy", 0.89, True),))
    ) == (UNSTABLE, UNSTABLE)
    assert decide(
        replace(_record(), control=(ControlEvidence("copy", 1.0, False),))
    ) == (UNSTABLE, UNSTABLE)


def test_harness_version_is_required_for_lookup() -> None:
    from nmesh.spec import best_for

    record = replace(_record(), harness="spec-v1")
    assert best_for(
        {"old": record}, record.target, record.spec, record.engine
    ) is None


def _arm(
    *,
    spec: SpecConfig | None = None,
    target: RoleIdentity | None = None,
    harness: str = SPEC_HARNESS_VERSION,
    rate: float = 10.0,
    rate_min: float | None = None,
    rate_max: float | None = None,
    content: str = "answer",
    unstable: bool = False,
) -> ArmRun:
    return ArmRun(
        target=target or RoleIdentity("target", "q4_k_m", "llamacpp", "artifact"),
        spec=spec or SpecConfig(),
        classes=(
            ClassResult(
                name="code",
                completion_tokens=10,
                decode_tps=rate,
                seconds=1.0,
                content_sha256=content,
                unstable=unstable,
                decode_tps_min=rate if rate_min is None else rate_min,
                decode_tps_max=rate if rate_max is None else rate_max,
                drafted=0,
                accepted=0,
            ),
        ),
        repeats=3,
        harness=harness,
        at=1.0,
    )


def test_control_validates_arms_and_calculates_ratio() -> None:
    first = _arm(rate=10.0)
    second = _arm(rate=5.0)
    assert control(first, second)[0].ratio == 0.5
    with pytest.raises(ValueError):
        control(first, _arm(spec=SpecConfig(kind=KIND_NGRAM)))
    with pytest.raises(ValueError):
        control(
            first,
            _arm(target=RoleIdentity("other", "q4_k_m", "llamacpp", "artifact")),
        )
    with pytest.raises(ValueError):
        control(first, _arm(harness="spec-v1"))


def test_run_arm_uses_medians_and_preserves_rate_extremes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter(
        [
            ("answer", 10, 1.0, 3.0, 0, 0),
            ("answer", 10, 100.0, 1.0, 0, 0),
            ("answer", 10, 3.0, 2.0, 0, 0),
        ]
    )
    monkeypatch.setattr(spec_measure, "_ask", lambda *args: next(samples))
    result = spec_measure.run_arm(
        None,
        "http://unused",
        "model",
        target=RoleIdentity("target", "q4_k_m", "llamacpp", "artifact"),
        spec=SpecConfig(),
        repeats=3,
        workloads=(Workload("code", "", 1),),
    )
    item = result.classes[0]
    assert item.decode_tps == 3.0
    assert item.seconds == 2.0
    assert item.decode_tps_min == 1.0
    assert item.decode_tps_max == 100.0


def test_run_arm_rejects_fewer_than_three_repeats() -> None:
    with pytest.raises(ValueError, match="three"):
        spec_measure.run_arm(
            None,
            "http://unused",
            "model",
            target=RoleIdentity("target", "q4_k_m", "llamacpp", "artifact"),
            spec=SpecConfig(),
            repeats=2,
            workloads=(Workload("code", "", 1),),
        )


def test_from_arms_carries_rate_spreads() -> None:
    reference = _arm(rate=10.0, rate_min=9.0, rate_max=11.0)
    candidate = _arm(
        spec=SpecConfig(kind=KIND_NGRAM),
        rate=12.0,
        rate_min=10.0,
        rate_max=14.0,
    )
    control_arm = _arm(rate=10.5, rate_min=10.0, rate_max=11.0)
    record = from_arms(
        reference,
        candidate,
        engine="engine",
        control_arm=control_arm,
        reference_id="ref",
        reference_tps=42.0,
        epoch="healthy",
    )
    evidence = record.classes[0]
    assert evidence.reference_spread == 0.2
    assert evidence.candidate_spread == (4.0 / 12.0)
    assert record.reference_id == "ref"
    assert record.reference_tps == 42.0
    assert record.epoch == "healthy"


def test_spec_decision_table_includes_mixed_regression() -> None:
    assert decide(_record()) == (ALLOW, ALLOW)
    assert decide(_record(identical=False)) == (NOT_IDENTICAL, NOT_IDENTICAL)
    assert decide(_record(speedup=1.0)) == (NOT_FASTER, NOT_FASTER)
    mixed = SpecRecord(
        target=_record().target,
        spec=_record().spec,
        engine="llama.cpp",
        harness=SPEC_HARNESS_VERSION,
        repeats=3,
        classes=(
            ClassEvidence("copy", 1.2, True, 10.0, 12.0, 1.0),
            ClassEvidence("prose", 0.9, True, 10.0, 9.0, 1.0),
        ),
        control=(ControlEvidence("copy", 1.0, True),),
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
    control_ratio: float = 1.0,
    control_identical: bool = True,
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
            control=(ControlEvidence("copy", control_ratio, control_identical),),
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


def test_spec_control_failure_refuses_with_unstable_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _planned_spec(
        tmp_path, monkeypatch, decision="allow", control_ratio=0.5
    )
    assert "--spec-type" not in result.services[0].launch.argv
    assert any("unstable" in warning for warning in result.warnings)


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
    assert cli.main(["spec", "measure", "--kind", "ngram", "--repeats", "2"]) == 2
    assert "repeats" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["spec", "measure", "--kind", "unknown"])


def test_spec_cli_no_reference_records_unknown_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    plan = _planned_spec(tmp_path, monkeypatch, decision=None)
    arms = iter(
        (
            _arm(rate=10.0),
            _arm(
                spec=SpecConfig(kind=KIND_NGRAM, n_max=3),
                rate=12.0,
            ),
            _arm(rate=10.0),
        )
    )

    class FakeSupervisor:
        def __init__(self, *, state_path: Path) -> None:
            self.state_path = state_path

        def up(self, *_args, **_kwargs) -> None:
            return None

        def down(self) -> None:
            return None

    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(
        cli,
        "_reference_context",
        lambda _service: (_ for _ in ()).throw(
            AssertionError("reference should be disabled")
        ),
    )
    monkeypatch.setattr(cli, "run_arm", lambda *_args, **_kwargs: next(arms))
    monkeypatch.setattr(cli, "engine_identity", lambda _profile: "engine")
    monkeypatch.setattr(cli.engine_runtime, "active", lambda: None)
    monkeypatch.setattr(cli, "service_fingerprint", lambda *_args: "artifact")

    assert cli.main([
        "spec", "measure", "--kind", "ngram", "--no-reference", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["epoch"] == "unknown"
    assert result["reference_id"] == ""
    assert result["reference_tps"] is None
    assert result["demoted"] == 0

    records = load_cache(tmp_path / "spec.json")
    assert next(iter(records.values())).epoch == "unknown"


def test_spec_cli_transport_failure_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    plan = _planned_spec(tmp_path, monkeypatch, decision=None)

    class FakeSupervisor:
        def __init__(self, *, state_path: Path) -> None:
            self.state_path = state_path

        def up(self, *_args, **_kwargs) -> None:
            return None

        def down(self) -> None:
            return None

    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(cli, "_reference_context", lambda _service: None)
    monkeypatch.setattr(cli, "engine_identity", lambda _profile: "engine")
    monkeypatch.setattr(cli.engine_runtime, "active", lambda: None)
    monkeypatch.setattr(cli, "service_fingerprint", lambda *_args: "artifact")
    monkeypatch.setattr(
        cli,
        "run_arm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.ReadTimeout("timed out")
        ),
    )

    assert cli.main([
        "spec", "measure", "--kind", "ngram", "--no-reference",
    ]) == 1
    assert "transport" in capsys.readouterr().err.lower()
    assert not (tmp_path / "spec.json").exists()
