"""Shared selection of evaluation and context evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nmesh.eval import SUITES, needle_tasks, suite_digest
from nmesh.eval.cache import EvalRecord
from nmesh.eval.context import ContextRecord


def valid_eval_records(
    records: Mapping[str, EvalRecord],
) -> tuple[
    dict[tuple[str, str, str, str, str, int, bool | None, int], EvalRecord],
    list[EvalRecord],
]:
    valid: dict[
        tuple[str, str, str, str, str, int, bool | None, int], EvalRecord
    ] = {}
    stale: list[EvalRecord] = []
    for record in records.values():
        tasks = SUITES.get(record.suite)
        if tasks is None or record.digest != suite_digest(tasks):
            stale.append(record)
            continue
        key = (
            record.model_id,
            record.quant,
            record.backend,
            record.suite,
            record.digest,
            record.reasoning_allowance,
            record.cache_prompt,
            record.depth,
        )
        previous = valid.get(key)
        if previous is None or record.at > previous.at:
            valid[key] = record
    return valid, stale


def planner_eval_records(
    records: Mapping[str, EvalRecord],
) -> dict[tuple[str, str, str], EvalRecord]:
    usable = {
        key: record
        for key, record in records.items()
        if not (record.unscorable or record.transport_errors or record.depth > 0)
    }
    valid, _ = valid_eval_records(usable)
    latest: dict[tuple[str, str, str], EvalRecord] = {}
    for record in valid.values():
        key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        previous = latest.get(key)
        if previous is None or record.at > previous.at:
            latest[key] = record
    return latest


def effective_context_records(
    records: Mapping[str, ContextRecord],
) -> tuple[ContextRecord, ...]:
    latest: dict[tuple[str, str, str, int], ContextRecord] = {}
    for record in records.values():
        if record.requested_depth <= 0:
            continue
        if record.probe_digest != suite_digest(
            needle_tasks(record.requested_depth, record.seed)
        ):
            continue
        key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
            record.requested_depth,
        )
        previous = latest.get(key)
        if previous is None or record.at > previous.at:
            latest[key] = record
    return tuple(latest.values())


@dataclass(frozen=True)
class DepthEvidence:
    verified: int
    lost: int


def context_depth_evidence(
    records: Mapping[str, ContextRecord],
) -> dict[tuple[str, str, str], DepthEvidence]:
    evidence: dict[tuple[str, str, str], DepthEvidence] = {}
    for record in effective_context_records(records):
        key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        served = record.served_depth or record.requested_depth
        current = evidence.get(key, DepthEvidence(0, 0))
        if any(family.lost for family in record.families):
            lost = served if current.lost == 0 else min(current.lost, served)
            evidence[key] = DepthEvidence(current.verified, lost)
        elif any(family.attributable for family in record.families):
            evidence[key] = DepthEvidence(max(current.verified, served), current.lost)
    return evidence
