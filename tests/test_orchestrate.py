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
    UNCONFIRMED,
    UNSTABLE,
    UNVERIFIED,
    Endpoint,
    RoleIdentity,
    best_for,
    combine,
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


def test_complete_only_sends_cache_prompt_when_configured() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    class Client:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def post(self, _url: str, *, json: dict[str, object]) -> Response:
            self.payloads.append(json)
            return Response()

    client = Client()
    protocol_module.complete(client, Endpoint("url", "model"), "prompt", 4)
    protocol_module.complete(
        client, Endpoint("url", "model", cache_prompt=False), "prompt", 4,
    )
    assert "cache_prompt" not in client.payloads[0]
    assert client.payloads[1]["cache_prompt"] is False


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


def _rows(*outcomes: tuple[str, bool, bool, bool, bool]) -> list[measure_module.TaskRow]:
    return [
        measure_module.TaskRow(
            id=task_id,
            category="hard",
            worker_passed=worker,
            lead_passed=lead,
            delegated_passed=delegated,
            accepted=accepted,
            unparsed_verdict=False,
            unscorable=False,
        )
        for task_id, worker, lead, delegated, accepted in outcomes
    ]


def test_combine_preserves_worst_quality_block_and_aggregates_timings() -> None:
    first = replace(
        _record_run(),
        delegated_passed=2,
        ceiling_passed=2,
        delegated_vs_lead=measure_module.Comparison(1, 0, 0.01),
        seconds_solo=1.0,
        seconds_delegated=4.0,
        at=2.0,
        rows=_rows(
            ("one", True, False, True, True),
            ("two", False, True, True, False),
        ),
    )
    second = replace(
        first,
        delegated_passed=1,
        ceiling_passed=1,
        delegated_vs_lead=measure_module.Comparison(0, 1, 0.9),
        seconds_solo=3.0,
        seconds_delegated=2.0,
        at=4.0,
        rows=_rows(
            ("one", True, False, False, True),
            ("two", False, True, True, False),
        ),
    )
    combined = combine((first, second))
    assert combined.delegated_passed == 1
    assert combined.ceiling_passed == 1
    assert combined.delegated_vs_lead.p == 0.9
    assert combined.seconds_solo == 2.0
    assert combined.seconds_delegated == 3.0
    assert combined.at == 4.0
    assert combined.repeats == 2
    assert combined.unstable_tasks == 1
    assert combined.rows == second.rows


def test_combine_identical_runs_have_no_unstable_tasks() -> None:
    run = replace(
        _record_run(),
        rows=_rows(
            ("one", True, False, True, True),
            ("two", False, True, True, False),
        ),
    )
    combined = combine((run, replace(run, at=3.0)))
    assert combined.unstable_tasks == 0


def test_combine_rejects_empty_mismatched_identity_and_task_ids() -> None:
    with pytest.raises(ValueError):
        combine(())
    run = replace(
        _record_run(),
        rows=_rows(("one", True, False, True, True)),
    )
    with pytest.raises(ValueError):
        combine((run, replace(run, suite="core")))
    with pytest.raises(ValueError):
        combine((run, replace(run, rows=_rows(("two", True, False, True, True)))))


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
    assert decide(record) == (UNCONFIRMED, UNCONFIRMED)
    assert decide(replace(record, repeats=2)) == (ALLOW, ALLOW)
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
    record = from_run(
        replace(_record_run(), repeats=2),
        reference_id="ref",
        reference_tps=50.0,
    )
    assert decide(record) == (ALLOW, ALLOW)
    assert decide(replace(record, epoch="healthy")) == (ALLOW, ALLOW)
    assert decide(replace(record, epoch="degraded")) == (ALLOW, ALLOW)
    assert decide(replace(record, repeats=1)) == (UNCONFIRMED, UNCONFIRMED)


def test_quality_gate_rejects_unstable_superior_evidence() -> None:
    record = from_run(replace(_record_run(), repeats=2, unstable_tasks=1))
    assert decide(record) == (UNSTABLE, UNSTABLE)
    assert decide(replace(record, unstable_tasks=0)) == (ALLOW, ALLOW)
    assert decide(replace(record, repeats=1, unstable_tasks=0)) == (
        UNCONFIRMED, UNCONFIRMED
    )
    not_superior = replace(
        record,
        delegated_passed=1,
        delegated_p=0.9,
        unstable_tasks=1,
    )
    assert decide(not_superior) == (NOT_SUPERIOR, NOT_SUPERIOR)


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


def test_delegation_demotion_accepts_exact_epoch_boundary() -> None:
    record = from_run(
        _record_run(),
        reference_id="ref",
        reference_tps=20.0,
        epoch="healthy",
    )
    records = {"boundary": record}
    assert demote_stale(records, "ref", 20.0 / EPOCH_MIN_RATIO) == (
        "boundary",
    )
    assert records["boundary"] == replace(record, epoch="degraded")


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
        ("repeats", 0),
        ("repeats", True),
        ("repeats", "2"),
        ("unstable_tasks", -1),
        ("unstable_tasks", True),
        ("unstable_tasks", "1"),
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
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(
        cli, "_orchestration_identity",
        lambda item: RoleIdentity(item.model_id, item.quant, item.backend),
    )
    def fake_measure(*args: object, **kwargs: object) -> measure_module.DelegationRun:
        del args
        calls.append(kwargs)
        return _record_run()

    monkeypatch.setattr(cli, "orchestrate_measure", fake_measure)
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
    assert result["repeats"] == 2
    assert result["unstable_tasks"] == 0
    assert len(calls) == 2
    assert saved["reference_tps"] == 0.0
    assert all(
        endpoint.cache_prompt is False
        for call in calls
        for endpoint in (call["lead"], call["worker"])
    )


def test_orchestrate_measure_leaves_cache_prompt_unset_for_other_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lead = SimpleNamespace(
        name="lead", model_id="lead", quant="q4", backend="llamacpp",
        model_ref="lead.gguf", roles=("chat",), port=1,
    )
    worker = SimpleNamespace(
        name="worker", model_id="worker", quant="q4", backend="ollama",
        model_ref="worker", roles=("worker",), port=2,
    )
    plan = SimpleNamespace(
        services=[lead, worker],
        routing=SimpleNamespace(role_to_service={"chat": "lead", "worker": "worker"}),
        profile=SimpleNamespace(),
    )
    endpoints: list[tuple[Endpoint, Endpoint]] = []
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(
        cli, "_orchestration_identity",
        lambda item: RoleIdentity(item.model_id, item.quant, item.backend),
    )
    monkeypatch.setattr(
        cli,
        "orchestrate_measure",
        lambda _tasks, **kwargs: (
            endpoints.append((kwargs["lead"], kwargs["worker"])) or _record_run()
        ),
    )
    monkeypatch.setattr(cli, "save_delegation", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_delegation_cache", dict)
    assert cli.main([
        "orchestrate", "measure", "--lead-url", "lead", "--worker-url",
        "worker", "--no-reference", "--repeats", "1",
    ]) == 0
    assert len(endpoints) == 1
    assert endpoints[0][0].cache_prompt is False
    assert endpoints[0][1].cache_prompt is None


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


def test_orchestrate_measure_demotion_write_failure_keeps_success(
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
    old_record = from_run(
        _record_run(),
        reference_id="ref",
        reference_tps=20.0,
        epoch="healthy",
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(
        cli, "_orchestration_identity",
        lambda item: RoleIdentity(item.model_id, item.quant, item.backend),
    )
    monkeypatch.setattr(cli, "orchestrate_measure", lambda *args, **kwargs: _record_run())
    monkeypatch.setattr(cli, "save_delegation", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_delegation_cache", lambda: {"old": old_record})
    monkeypatch.setattr(
        cli, "_reference_context",
        lambda _: (Path("llama-bench"), Path("reference.gguf"), "ref", 4),
    )
    monkeypatch.setattr(cli, "load_history", dict)
    monkeypatch.setattr(cli, "save_history", lambda _history: None)
    monkeypatch.setattr(cli, "measure_reference", lambda *_args: 50.0)
    monkeypatch.setattr(
        cli,
        "save_all_delegation",
        lambda _records: (_ for _ in ()).throw(OSError("readonly")),
    )
    assert cli.main([
        "orchestrate", "measure", "--lead-url", "lead", "--worker-url",
        "worker", "--json",
    ]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["demoted"] == 1
    assert result["gate"] == ALLOW
    assert "readonly" in captured.err
