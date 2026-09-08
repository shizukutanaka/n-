from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import nmesh.orchestrate.protocol as protocol_module
from nmesh import cli
from nmesh.bench import EPOCH_MIN_RATIO, EpochSample
from nmesh.eval import Task
from nmesh.orchestrate import (
    ALLOW,
    CHEAPER,
    COSTLIER,
    NO_EVIDENCE,
    NOT_SUPERIOR,
    PROTOCOL_VERSION,
    STALE,
    UNVERIFIED,
    Endpoint,
    RoleIdentity,
    best_for,
    decide,
    decide_cost,
    delegate,
    demote_stale,
    from_run,
    load_cache,
    measure,
    read_verdict,
    save,
    save_all,
)
from nmesh.orchestrate.protocol import Call

measure_module = importlib.import_module("nmesh.orchestrate.measure")


def _record_run() -> measure_module.DelegationRun:
    lead = RoleIdentity("lead", "q4", "llamacpp")
    worker = RoleIdentity("worker", "q4", "llamacpp")
    return measure_module.DelegationRun(
        lead=lead,
        worker=worker,
        suite="hard",
        digest="digest",
        n_tasks=2,
        worker_passed=1,
        lead_passed=1,
        delegated_passed=2,
        ceiling_passed=2,
        delegated_vs_lead=measure_module.Comparison(1, 0, 0.01),
        ceiling_vs_lead=measure_module.Comparison(1, 0, 0.01),
        verifier=measure_module.VerifierReport(2, 0, 0, 0, 1.0),
        lead_tokens_solo=2101,
        lead_tokens_delegated=5084,
        seconds_solo=7.080,
        seconds_delegated=16.608,
        unscorable=0,
        reasoning_allowance=0,
        protocol=PROTOCOL_VERSION,
        at=1.0,
    )


def test_read_verdict_accepts_unambiguous_verdicts() -> None:
    assert read_verdict("YES") is True
    assert read_verdict("YES.") is True
    assert read_verdict(" yes\n") is True
    assert read_verdict("NO") is False
    assert read_verdict("NO, the format is wrong") is False
    assert read_verdict("Maybe yes, maybe no") is None
    assert read_verdict("") is None


def test_orchestrate_show_json_handles_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_delegation_cache", dict)
    assert cli.main(["orchestrate", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_delegate_escalates_once_on_unreadable_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_complete(_client: object, endpoint: Endpoint, prompt: str, max_tokens: int) -> Call:
        del max_tokens
        calls.append(endpoint.model_ref)
        if endpoint.model_ref == "verify":
            return Call("maybe", 2, 1, False, 0.0)
        return Call("rescued", 2, 1, False, 0.0)

    monkeypatch.setattr(protocol_module, "complete", fake_complete)
    result = delegate(
        object(), "task", 10, lead=Endpoint("lead", "lead"), worker=Endpoint("worker", "worker")
    )
    assert result.escalated
    assert result.unparsed_verdict
    assert calls == ["worker", "lead", "lead"]


def test_measure_reports_paired_arms_and_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = (
        Task("one", "test", "one", 8, lambda text: text == "good"),
        Task("two", "test", "two", 8, lambda text: text == "good"),
    )

    def fake_complete(_client: object, endpoint: Endpoint, prompt: str, max_tokens: int) -> Call:
        del max_tokens
        if endpoint.model_ref == "worker":
            text = "good" if prompt == "one" else "bad"
        elif prompt.startswith("You are a strict checker"):
            text = "YES" if "ANSWER:\ngood" in prompt else "NO"
        else:
            text = "bad" if prompt == "one" else "good"
        return Call(text, 3, 2, False, 0.1)

    monkeypatch.setattr(measure_module, "complete", fake_complete)
    monkeypatch.setattr(protocol_module, "complete", fake_complete)
    run = measure(
        tasks,
        lead=Endpoint("lead", "lead"),
        worker=Endpoint("worker", "worker"),
        lead_identity=RoleIdentity("lead", "q4", "llamacpp"),
        worker_identity=RoleIdentity("worker", "q4", "llamacpp"),
    )
    assert (run.worker_passed, run.lead_passed, run.delegated_passed) == (1, 1, 2)
    assert run.ceiling_passed == 2
    assert (run.delegated_vs_lead.gained, run.delegated_vs_lead.lost) == (1, 0)
    assert run.lead_tokens_solo == 10
    assert run.lead_tokens_delegated == 15
    assert run.verifier.accepted_but_wrong == 0
    assert run.verifier.rejected_but_right == 0


def test_record_round_trip_and_gate(tmp_path: Path) -> None:
    lead = RoleIdentity("lead", "q4", "llamacpp")
    worker = RoleIdentity("worker", "q4", "llamacpp")
    run = measure_module.DelegationRun(
        lead=lead,
        worker=worker,
        suite="hard",
        digest="digest",
        n_tasks=2,
        worker_passed=1,
        lead_passed=1,
        delegated_passed=2,
        ceiling_passed=2,
        delegated_vs_lead=measure_module.Comparison(1, 0, 0.01),
        ceiling_vs_lead=measure_module.Comparison(1, 0, 0.01),
        verifier=measure_module.VerifierReport(2, 0, 0, 0, 1.0),
        lead_tokens_solo=2,
        lead_tokens_delegated=4,
        seconds_solo=1.0,
        seconds_delegated=2.0,
        unscorable=0,
        reasoning_allowance=0,
        protocol=PROTOCOL_VERSION,
        at=1.0,
    )
    path = tmp_path / "delegation.json"
    save(run, path)
    cache = load_cache(path)
    record = next(iter(cache.values()))
    assert decide(record) == (ALLOW, ALLOW)
    assert best_for(cache, lead, worker, PROTOCOL_VERSION) == record
    path.write_text("{", encoding="utf-8")
    assert load_cache(path) == {}
    path.write_text(json.dumps({"results": {"bad": {"n_tasks": 0}}}), encoding="utf-8")
    assert load_cache(path) == {}


def test_gate_rejects_non_superior_records(tmp_path: Path) -> None:
    lead = RoleIdentity("lead", "q4", "llamacpp")
    worker = RoleIdentity("worker", "q4", "llamacpp")
    run = measure_module.DelegationRun(
        lead=lead, worker=worker, suite="hard", digest="digest", n_tasks=2,
        worker_passed=1, lead_passed=2, delegated_passed=1, ceiling_passed=2,
        delegated_vs_lead=measure_module.Comparison(0, 1, 0.9),
        ceiling_vs_lead=measure_module.Comparison(0, 1, 0.9),
        verifier=measure_module.VerifierReport(2, 0, 0, 0, 1.0),
        lead_tokens_solo=2, lead_tokens_delegated=4, seconds_solo=1.0,
        seconds_delegated=2.0, unscorable=0, reasoning_allowance=0,
        protocol=PROTOCOL_VERSION, at=1.0,
    )
    record = from_run(run)
    assert decide(record) == (NOT_SUPERIOR, NOT_SUPERIOR)
    assert best_for({record.digest: record}, lead, worker, PROTOCOL_VERSION) == record
    assert decide(None) == (NO_EVIDENCE, NO_EVIDENCE)


def test_quality_gate_is_independent_of_host_epoch() -> None:
    record = from_run(_record_run(), reference_id="ref", reference_tps=50.0)
    assert decide(record) == (ALLOW, ALLOW)
    assert decide(replace(record, epoch="healthy")) == (ALLOW, ALLOW)
    assert decide(replace(record, epoch="degraded")) == (ALLOW, ALLOW)


def test_cost_decision_and_ratios() -> None:
    record = from_run(_record_run(), reference_id="ref", reference_tps=50.0)
    assert decide_cost(None) == (NO_EVIDENCE, NO_EVIDENCE)
    assert decide_cost(replace(record, seconds_solo=0.0)) == (
        NO_EVIDENCE, NO_EVIDENCE
    )
    assert decide_cost(from_run(_record_run())) == (UNVERIFIED, UNVERIFIED)
    assert decide_cost(replace(record, epoch="degraded")) == (STALE, STALE)
    assert decide_cost(replace(record, seconds_delegated=6.0)) == (
        CHEAPER, CHEAPER
    )
    assert decide_cost(record) == (COSTLIER, COSTLIER)
    assert record.seconds_ratio == pytest.approx(16.608 / 7.080)
    assert record.token_ratio == pytest.approx(5084 / 2101)
    assert replace(record, lead_tokens_solo=0).token_ratio == 0.0


def test_delegation_demotion_preserves_quality_fields() -> None:
    record = from_run(
        _record_run(),
        reference_id="ref",
        reference_tps=20.0,
        epoch="healthy",
    )
    records = {"key": record}
    assert 48.0 / 20.0 >= 1 / EPOCH_MIN_RATIO
    assert demote_stale(records, "ref", 48.0) == ("key",)
    assert records["key"] == replace(record, epoch="degraded")
    assert demote_stale(records, "other", 100.0) == ()
    assert demote_stale(
        {"legacy": from_run(_record_run())}, "ref", 100.0
    ) == ()


def test_delegation_record_round_trip_defaults_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "delegation.json"
    record = from_run(
        _record_run(),
        reference_id="ref",
        reference_tps=40.0,
        epoch="healthy",
    )
    save_all({"key": record}, path)
    assert load_cache(path)["key"] == record
    payload = json.loads(path.read_text(encoding="utf-8"))
    legacy = dict(payload["results"]["key"])
    for name in ("reference_id", "reference_tps", "epoch"):
        legacy.pop(name)
    path.write_text(json.dumps({"results": {"key": legacy}}), encoding="utf-8")
    loaded = load_cache(path)["key"]
    assert (loaded.reference_id, loaded.reference_tps, loaded.epoch) == (
        "", 0.0, "unknown"
    )
    for field, value in (
        ("epoch", "invalid"),
        ("reference_id", 1),
        ("reference_tps", -1.0),
        ("reference_tps", float("nan")),
    ):
        invalid = dict(legacy)
        invalid[field] = value
        path.write_text(
            json.dumps({"results": {"key": invalid}}),
            encoding="utf-8",
        )
        assert load_cache(path) == {}


def test_orchestrate_measure_no_reference_reports_unverified_cost(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = SimpleNamespace(
        name="lead", model_id="lead", quant="q4", backend="llamacpp",
        model_ref="lead.gguf", roles=("chat",), port=1,
    )
    worker = SimpleNamespace(
        name="worker", model_id="worker", quant="q4", backend="llamacpp",
        model_ref="worker.gguf", roles=("worker",), port=2,
    )
    plan = SimpleNamespace(
        services=[service, worker],
        routing=SimpleNamespace(role_to_service={"chat": "lead", "worker": "worker"}),
        profile=SimpleNamespace(),
    )
    saved: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(
        cli, "_orchestration_identity",
        lambda item: RoleIdentity(item.model_id, item.quant, item.backend),
    )
    monkeypatch.setattr(cli, "orchestrate_measure", lambda *args, **kwargs: _record_run())
    monkeypatch.setattr(
        cli, "save_delegation",
        lambda run, **kwargs: saved.update({"run": run, **kwargs}),
    )
    monkeypatch.setattr(cli, "load_delegation_cache", dict)
    monkeypatch.setattr(cli, "_reference_context", lambda _: pytest.fail("used"))
    assert cli.main([
        "orchestrate", "measure", "--lead-url", "lead", "--worker-url",
        "worker", "--no-reference", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cost"] == UNVERIFIED
    assert result["reference_id"] == ""
    assert saved["reference_tps"] == 0.0


def test_orchestrate_measure_degraded_epoch_stales_only_cost(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = SimpleNamespace(
        name="lead", model_id="lead", quant="q4", backend="llamacpp",
        model_ref="lead.gguf", roles=("chat",), port=1,
    )
    worker = SimpleNamespace(
        name="worker", model_id="worker", quant="q4", backend="llamacpp",
        model_ref="worker.gguf", roles=("worker",), port=2,
    )
    plan = SimpleNamespace(
        services=[service, worker],
        routing=SimpleNamespace(role_to_service={"chat": "lead", "worker": "worker"}),
        profile=SimpleNamespace(),
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(
        cli, "_orchestration_identity",
        lambda item: RoleIdentity(item.model_id, item.quant, item.backend),
    )
    monkeypatch.setattr(cli, "orchestrate_measure", lambda *args, **kwargs: _record_run())
    monkeypatch.setattr(cli, "save_delegation", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_delegation_cache", dict)
    monkeypatch.setattr(
        cli, "_reference_context",
        lambda _: (Path("llama-bench"), Path("reference.gguf"), "ref", 4),
    )
    monkeypatch.setattr(
        cli, "load_history",
        lambda: {"ref": (EpochSample("ref", 100.0, "old"),)},
    )
    monkeypatch.setattr(cli, "save_history", lambda _history: None)
    monkeypatch.setattr(cli, "measure_reference", lambda *_args: 50.0)
    assert cli.main([
        "orchestrate", "measure", "--lead-url", "lead", "--worker-url",
        "worker", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["epoch"] == "degraded"
    assert result["cost"] == STALE
    assert result["gate"] == ALLOW
