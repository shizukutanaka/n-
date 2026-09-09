"""Shared evidence predicates and saved-evidence inventory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nmesh.eval.cache import EvalRecord
    from nmesh.eval.context import ContextRecord

EPOCH_MIN_RATIO = 0.80


def refutes(recorded_tps: float, reference_tps: float) -> bool:
    """Return whether a faster reference disproves recorded throughput."""
    return (
        math.isfinite(recorded_tps)
        and recorded_tps > 0
        and math.isfinite(reference_tps)
        and reference_tps > 0
        and reference_tps >= recorded_tps / EPOCH_MIN_RATIO
    )


def _bench_parts(key: str) -> tuple[str, str, str]:
    parts = key.split("|")
    return tuple(parts[:3]) if len(parts) >= 3 else (key, "", "")


def _bench_rows() -> list[dict[str, object]]:
    from nmesh.bench.cache import (
        BENCH_HARNESS_VERSION,
        load_cache,
        load_records,
    )

    records = load_records()
    usable_keys = set(load_cache())
    rows: list[dict[str, object]] = []
    for key, record in records.items():
        model_id, quant, backend = _bench_parts(key)
        reasons: list[str] = []
        if record.harness != BENCH_HARNESS_VERSION:
            reasons.append("harness_mismatch")
        if not record.stable:
            reasons.append("unstable")
        if record.epoch == "degraded":
            reasons.append("epoch_degraded")
        if record.confirmations < 2:
            reasons.append("unconfirmed")
        rows.append({
            "kind": "bench",
            "key": key,
            "model_id": model_id,
            "quant": quant,
            "backend": backend,
            "value": record.tps,
            "usable": key in usable_keys,
            "reasons": reasons,
            "remeasure": "nmesh bench" if reasons else "",
        })
    return rows


def _eval_rows(records: Mapping[str, EvalRecord]) -> list[dict[str, object]]:
    from nmesh import cli
    from nmesh.eval import SUITES, suite_digest

    planner_records = cli._eval_planner_records(records)
    rows: list[dict[str, object]] = []
    for key, record in records.items():
        reasons: list[str] = []
        tasks = SUITES.get(record.suite)
        if tasks is None:
            reasons.append("suite_unknown")
        elif record.digest != suite_digest(tasks):
            reasons.append("grader_digest_mismatch")
        if record.unscorable:
            reasons.append("unscorable")
        if record.transport_errors:
            reasons.append("transport_errors")
        if record.depth > 0:
            reasons.append("depth_scoped")
        planner_key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        usable = planner_records.get(planner_key) is record
        rows.append({
            "kind": "eval",
            "key": key,
            "model_id": record.model_id,
            "quant": record.quant,
            "backend": record.backend,
            "suite": record.suite,
            "value": f"{record.passed}/{record.n_tasks}",
            "usable": usable,
            "reasons": reasons,
            "remeasure": (
                f"nmesh eval --suite {record.suite}" if reasons else ""
            ),
        })
    return rows


def _depth_rows(records: Mapping[str, ContextRecord]) -> list[dict[str, object]]:
    from nmesh import cli
    from nmesh.eval import needle_tasks, suite_digest

    effective = set(cli._context_records(records))
    rows: list[dict[str, object]] = []
    for key, record in records.items():
        reasons: list[str] = []
        expected = (
            suite_digest(needle_tasks(record.requested_depth, record.seed))
            if record.requested_depth > 0 else ""
        )
        if record.requested_depth <= 0 or record.probe_digest != expected:
            reasons.append("probe_digest_mismatch")
        if not any(family.attributable for family in record.families):
            reasons.append("control_failed")
        if any(family.lost for family in record.families):
            reasons.append("depth_lost")
        usable = not reasons and record in effective
        rows.append({
            "kind": "depth",
            "key": key,
            "model_id": record.model_id,
            "quant": record.quant,
            "backend": record.backend,
            "requested_depth": record.requested_depth,
            "served_depth": record.served_depth,
            "value": "verified" if usable else "lost",
            "usable": usable,
            "reasons": reasons,
            "remeasure": (
                f"nmesh eval --depth {record.requested_depth}"
                if reasons else ""
            ),
        })
    return rows


def collect_evidence() -> dict[str, object]:
    from nmesh.eval.cache import load_eval_cache
    from nmesh.eval.context import load_context_cache

    records = (
        _bench_rows()
        + _eval_rows(load_eval_cache())
        + _depth_rows(load_context_cache())
    )
    counts = {
        "total": len(records),
        "usable": sum(bool(row["usable"]) for row in records),
        "bench": sum(row["kind"] == "bench" for row in records),
        "eval": sum(row["kind"] == "eval" for row in records),
        "depth": sum(row["kind"] == "depth" for row in records),
    }
    return {"records": records, "counts": counts}
