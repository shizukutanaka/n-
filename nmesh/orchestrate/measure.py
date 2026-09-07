"""Falsify a lead/worker delegation before it is allowed to serve traffic.

Delegation is only worth its extra calls if the pair beats the lead model
alone. This module measures four things on the same suite, same binaries and
same host, at temperature 0:

``worker``
    the small model answering every task by itself,
``lead``
    the strong model answering every task by itself,
``delegated``
    the bounded delegate/verify/escalate protocol,
``ceiling``
    what delegation would score if the verifier were perfect (accept exactly
    the worker answers that pass, escalate the rest).

The ceiling is the part that decides whether the idea can pay at all: it is a
property of the two models' disagreement, not of the prompt, and it is
measured without spending a single verifier call. If the ceiling is not above
the lead's own score by more than the paired test can resolve, no verifier
prompt can rescue the arrangement.

Grading is done by the suite's own deterministic verifiers; no model grades
another model's output on the scoring path.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import httpx

from nmesh.eval import Task, suite_digest
from nmesh.eval.stats import mcnemar_two_sided

from .protocol import (
    PROTOCOL_VERSION,
    Cost,
    Endpoint,
    Ledger,
    complete,
    delegate,
)


@dataclass(frozen=True)
class RoleIdentity:
    """What was actually running behind one role."""

    model_id: str
    quant: str
    backend: str
    artifact: str = ""


@dataclass(frozen=True)
class TaskRow:
    """Per-task evidence, kept so a disagreement can be inspected later."""

    id: str
    category: str
    worker_passed: bool
    lead_passed: bool
    delegated_passed: bool
    accepted: bool
    unparsed_verdict: bool
    unscorable: bool


@dataclass(frozen=True)
class Comparison:
    """One paired comparison against the lead-alone arm."""

    gained: int
    lost: int
    p: float


@dataclass(frozen=True)
class VerifierReport:
    """How well the lead judged the worker's answers."""

    accepted: int
    accepted_but_wrong: int
    rejected_but_right: int
    unparsed: int
    accuracy: float


@dataclass(frozen=True)
class DelegationRun:
    """A complete, self-describing delegation measurement."""

    lead: RoleIdentity
    worker: RoleIdentity
    suite: str
    digest: str
    n_tasks: int
    worker_passed: int
    lead_passed: int
    delegated_passed: int
    ceiling_passed: int
    delegated_vs_lead: Comparison
    ceiling_vs_lead: Comparison
    verifier: VerifierReport
    lead_tokens_solo: int
    lead_tokens_delegated: int
    seconds_solo: float
    seconds_delegated: float
    unscorable: int
    reasoning_allowance: int
    protocol: str
    at: float
    rows: list[TaskRow] = field(default_factory=list)

    @property
    def verify_overhead(self) -> float:
        """Lead tokens spent per lead token that answering alone would cost.

        Above 1.0 the lead reads more to check the worker than it would to do
        the work itself, which is the structural reason short tasks cannot be
        delegated profitably.
        """
        if self.lead_tokens_solo <= 0:
            return 0.0
        return self.lead_tokens_delegated / self.lead_tokens_solo


def _passed(task: Task, text: str, unscorable: bool) -> bool:
    return False if unscorable else bool(task.check(text))


def _comparison(rows: Sequence[TaskRow], candidate: str) -> Comparison:
    gained = sum(
        1 for row in rows if getattr(row, candidate) and not row.lead_passed
    )
    lost = sum(
        1 for row in rows if row.lead_passed and not getattr(row, candidate)
    )
    return Comparison(gained, lost, mcnemar_two_sided(gained, lost))


def measure(
    tasks: Iterable[Task],
    *,
    lead: Endpoint,
    worker: Endpoint,
    lead_identity: RoleIdentity,
    worker_identity: RoleIdentity,
    suite: str = "hard",
    reasoning_allowance: int = 0,
    timeout: float = 300.0,
) -> DelegationRun:
    """Run all four arms over ``tasks`` and return the measurement."""
    items = list(tasks)
    rows: list[TaskRow] = []
    ledger = Ledger()
    solo = Cost()
    ceiling_flags: list[bool] = []
    transport_failures = 0
    with httpx.Client(timeout=timeout) as client:
        for task in items:
            budget = task.max_tokens + max(0, reasoning_allowance)
            try:
                lead_call = complete(client, lead, task.prompt, budget)
                solo.add(lead_call)
                outcome = delegate(
                    client,
                    task.prompt,
                    budget,
                    lead=lead,
                    worker=worker,
                    ledger=ledger,
                )
            except (httpx.HTTPError, ValueError):
                transport_failures += 1
                continue
            lead_passed = _passed(task, lead_call.text, lead_call.unscorable)
            worker_passed = _passed(
                task, outcome.worker.text, outcome.worker.unscorable
            )
            rows.append(
                TaskRow(
                    id=task.id,
                    category=task.category,
                    worker_passed=worker_passed,
                    lead_passed=lead_passed,
                    delegated_passed=_passed(
                        task, outcome.answer, outcome.unscorable
                    ),
                    accepted=outcome.accepted,
                    unparsed_verdict=outcome.unparsed_verdict,
                    unscorable=outcome.unscorable or lead_call.unscorable,
                )
            )
            ceiling_flags.append(worker_passed or lead_passed)
    if not rows:
        raise RuntimeError("no delegation task completed")
    if transport_failures:
        raise RuntimeError(
            f"{transport_failures} delegation tasks failed at transport level"
        )
    ceiling_rows = [
        TaskRow(
            id=row.id,
            category=row.category,
            worker_passed=row.worker_passed,
            lead_passed=row.lead_passed,
            delegated_passed=flag,
            accepted=row.accepted,
            unparsed_verdict=row.unparsed_verdict,
            unscorable=row.unscorable,
        )
        for row, flag in zip(rows, ceiling_flags)
    ]
    agreements = sum(1 for row in rows if row.accepted == row.worker_passed)
    return DelegationRun(
        lead=lead_identity,
        worker=worker_identity,
        suite=suite,
        digest=suite_digest(items),
        n_tasks=len(rows),
        worker_passed=sum(row.worker_passed for row in rows),
        lead_passed=sum(row.lead_passed for row in rows),
        delegated_passed=sum(row.delegated_passed for row in rows),
        ceiling_passed=sum(ceiling_flags),
        delegated_vs_lead=_comparison(rows, "delegated_passed"),
        ceiling_vs_lead=_comparison(ceiling_rows, "delegated_passed"),
        verifier=VerifierReport(
            accepted=sum(row.accepted for row in rows),
            accepted_but_wrong=sum(
                1 for row in rows if row.accepted and not row.worker_passed
            ),
            rejected_but_right=sum(
                1 for row in rows if not row.accepted and row.worker_passed
            ),
            unparsed=sum(row.unparsed_verdict for row in rows),
            accuracy=agreements / len(rows),
        ),
        lead_tokens_solo=solo.prompt_tokens + solo.completion_tokens,
        lead_tokens_delegated=(
            ledger.lead_prompt_tokens + ledger.lead_completion_tokens
        ),
        seconds_solo=solo.seconds,
        seconds_delegated=ledger.seconds,
        unscorable=sum(row.unscorable for row in rows),
        reasoning_allowance=reasoning_allowance,
        protocol=PROTOCOL_VERSION,
        at=time.time(),
        rows=rows,
    )
