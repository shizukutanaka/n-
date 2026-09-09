"""Inventory saved evidence without changing planner policy."""

from __future__ import annotations

from collections.abc import Mapping

from nmesh.bench.cache import (
    BENCH_HARNESS_VERSION,
    load_cache,
    load_records,
)
from nmesh.eval import SUITES, needle_tasks, suite_digest
from nmesh.eval.cache import EvalRecord, load_eval_cache
from nmesh.eval.context import ContextRecord, load_context_cache
from nmesh.eval.select import (
    DepthEvidence,
    context_depth_evidence,
    effective_context_records,
    planner_eval_records,
)


def _bench_parts(key: str) -> tuple[str, str, str]:
    parts = key.split("|")
    return tuple(parts[:3]) if len(parts) >= 3 else (key, "", "")


def _bench_rows() -> list[dict[str, object]]:
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
    planner_records = planner_eval_records(records)
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
        selected = planner_records.get(planner_key)
        usable = selected is record
        if not usable and not reasons and selected is not None:
            reasons.append("superseded")
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
                f"nmesh eval --suite {record.suite}"
                if reasons and reasons != ["superseded"] else ""
            ),
        })
    return rows


def _depth_value(
    record: ContextRecord,
    *,
    effective: bool,
    evidence: Mapping[tuple[str, str, str], DepthEvidence],
) -> str:
    served = record.served_depth or record.requested_depth
    families = " ".join(
        f"{family.name} {family.passed}/{family.of} "
        f"control {family.control_passed}/{family.control_of}"
        for family in record.families
    )
    value = (
        f"served={served}"
        if record.served_depth
        else f"requested={record.requested_depth}"
    )
    if families:
        value += f" {families}"
    verdict = _depth_verdict(record, effective=effective, evidence=evidence)
    if verdict:
        value += f" {verdict}"
    return value


def _depth_verdict(
    record: ContextRecord,
    *,
    effective: bool,
    evidence: Mapping[tuple[str, str, str], DepthEvidence],
) -> str:
    if not effective:
        return ""
    served = record.served_depth or record.requested_depth
    key = (
        record.model_id.casefold(),
        record.quant.casefold(),
        record.backend.casefold(),
    )
    depth_evidence = evidence.get(key)
    if depth_evidence is None:
        return ""
    if depth_evidence.lost >= served:
        return "lost"
    if depth_evidence.verified >= served:
        return "verified"
    return ""


def _depth_rows(records: Mapping[str, ContextRecord]) -> list[dict[str, object]]:
    effective_records = set(effective_context_records(records))
    evidence = context_depth_evidence(records)
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
        effective = record in effective_records
        if not effective and not reasons:
            reasons.append("superseded")
        usable = effective and not reasons
        verdict = _depth_verdict(record, effective=effective, evidence=evidence)
        rows.append({
            "kind": "depth",
            "key": key,
            "model_id": record.model_id,
            "quant": record.quant,
            "backend": record.backend,
            "requested_depth": record.requested_depth,
            "served_depth": record.served_depth,
            "value": _depth_value(
                record,
                effective=effective,
                evidence=evidence,
            ),
            "verdict": verdict,
            "usable": usable,
            "reasons": reasons,
            "remeasure": (
                f"nmesh eval --depth {record.requested_depth}"
                if reasons and reasons != ["superseded"] else ""
            ),
        })
    return rows


def collect_evidence() -> dict[str, object]:
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
