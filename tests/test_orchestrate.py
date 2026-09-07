from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import nmesh.orchestrate.protocol as protocol_module
from nmesh import cli
from nmesh.eval import Task
from nmesh.orchestrate import (
    ALLOW,
    NO_EVIDENCE,
    NOT_SUPERIOR,
    PROTOCOL_VERSION,
    Endpoint,
    RoleIdentity,
    best_for,
    decide,
    delegate,
    from_run,
    load_cache,
    measure,
    read_verdict,
    save,
)
from nmesh.orchestrate.protocol import Call

measure_module = importlib.import_module("nmesh.orchestrate.measure")


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
