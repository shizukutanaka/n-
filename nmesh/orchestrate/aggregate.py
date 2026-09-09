"""Aggregate repeated delegation measurements into one evidence record."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import replace

from .measure import DelegationRun

MIN_REPEATS = 2


def _identity(run: DelegationRun) -> tuple[object, ...]:
    return (
        run.lead,
        run.worker,
        run.suite,
        run.digest,
        run.n_tasks,
        run.protocol,
        run.reasoning_allowance,
    )


def _outcomes(run: DelegationRun) -> dict[str, tuple[bool, bool, bool, bool]]:
    outcomes: dict[str, tuple[bool, bool, bool, bool]] = {}
    for row in run.rows:
        if row.id in outcomes:
            raise ValueError("delegation runs contain duplicate task ids")
        outcomes[row.id] = (
            row.worker_passed,
            row.lead_passed,
            row.delegated_passed,
            row.accepted,
        )
    return outcomes


def combine(runs: Sequence[DelegationRun]) -> DelegationRun:
    """Combine repeated runs while preserving the worst quality evidence."""
    if not runs:
        raise ValueError("at least one delegation run is required")
    first = runs[0]
    identity = _identity(first)
    for run in runs[1:]:
        if _identity(run) != identity:
            raise ValueError("delegation runs disagree on their identity")

    outcome_maps = [_outcomes(run) for run in runs]
    task_ids = set(outcome_maps[0])
    if any(set(outcomes) != task_ids for outcomes in outcome_maps[1:]):
        raise ValueError("delegation runs disagree on task ids")
    unstable_tasks = sum(
        len({outcomes[task_id] for outcomes in outcome_maps}) > 1
        for task_id in task_ids
    )
    worst_index = min(
        range(len(runs)),
        key=lambda index: (
            runs[index].delegated_passed,
            runs[index].ceiling_passed,
            index,
        ),
    )
    worst = runs[worst_index]
    return replace(
        worst,
        seconds_solo=statistics.median(run.seconds_solo for run in runs),
        seconds_delegated=statistics.median(
            run.seconds_delegated for run in runs
        ),
        at=max(run.at for run in runs),
        rows=list(worst.rows),
        repeats=len(runs),
        unstable_tasks=unstable_tasks,
    )
