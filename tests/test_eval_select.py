"""Tests for nmesh/eval/select.py — planner-facing evidence selection."""

from __future__ import annotations

from nmesh.eval import SUITES, suite_digest
from nmesh.eval.cache import EvalRecord
from nmesh.eval.select import planner_eval_records, valid_eval_records


def _record(**overrides) -> EvalRecord:
    digest = suite_digest(SUITES["core"])
    base = {
        "model_id": "m",
        "quant": "q4_k_m",
        "backend": "llamacpp",
        "n_tasks": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "by_category": {},
        "at": 1.0,
        "suite": "core",
        "digest": digest,
    }
    base.update(overrides)
    return EvalRecord(**base)


def test_valid_eval_records_drops_stale_digest() -> None:
    fresh = _record()
    stale = _record(digest="v0:outdated")
    valid, dropped = valid_eval_records({"a": fresh, "b": stale})
    assert list(valid.values()) == [fresh]
    assert dropped == [stale]


def test_valid_eval_records_keeps_newest_per_key() -> None:
    older = _record(at=10.0)
    newer = _record(at=20.0)
    valid, _ = valid_eval_records({"a": older, "b": newer})
    assert len(valid) == 1
    assert next(iter(valid.values())).at == 20.0


def test_planner_eval_records_excludes_depth_and_failed_runs() -> None:
    good = _record(at=30.0)
    deep = _record(at=40.0, depth=8)
    broken = _record(at=50.0, transport_errors=1)
    unscorable = _record(at=60.0, unscorable=2)
    latest = planner_eval_records(
        {"a": good, "b": deep, "c": broken, "d": unscorable}
    )
    assert list(latest.values()) == [good]


def test_planner_eval_records_dedupes_case_insensitively() -> None:
    lower = _record(model_id="M", quant="Q4_K_M", backend="LLaMACPP", at=5.0)
    upper = _record(at=9.0)
    latest = planner_eval_records({"a": lower, "b": upper})
    assert len(latest) == 1
    assert next(iter(latest.values())).at == 9.0


def test_planner_eval_records_keeps_older_good_run_when_newest_is_bad() -> None:
    good = _record(at=30.0)
    failed_newer = _record(at=90.0, transport_errors=1)
    latest = planner_eval_records({"a": good, "b": failed_newer})
    assert list(latest.values()) == [good]
