from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass

import httpx

from .suite import Task


@dataclass(frozen=True)
class TaskOutcome:
    id: str
    category: str
    passed: bool
    output: str
    unscorable: bool = False
    value_passed: bool | None = None


@dataclass(frozen=True)
class EvalRun:
    model_id: str
    quant: str
    backend: str
    n_tasks: int
    passed: int
    pass_rate: float
    by_category: dict[str, float]
    outcomes: list[TaskOutcome]
    at: float
    artifact: str = ""
    suite: str = "core"
    digest: str = ""
    unscorable: int = 0
    reasoning_allowance: int = 0


def _output(value: object) -> str:
    return value if isinstance(value, str) else ""


def _error(error: Exception) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}"[:200]


def run(
    tasks: Iterable[Task],
    base_url: str,
    model_ref: str,
    *,
    timeout: float = 120.0,
    reasoning_allowance: int = 0,
) -> EvalRun:
    """Ask each task and grade the answer text.

    A task budget is an answer budget: the suite gives 16 to 48 tokens because
    that is what the answers need. A model that spends the budget on separate
    reasoning output returns an empty `content` with `finish_reason` of
    `length`, and grading that empty string measures the budget, not the model.
    Measured on this machine with gemma-4-26B-A4B-it-qat UD-Q4_K_XL: every one
    of the 104 extended tasks returned empty content at the suite budget, while
    `arithmetic.add`, `instruction.echo` and `format.json_city` all answered
    correctly at 512 tokens. Those answerless responses are counted as
    unscorable rather than failed, and `reasoning_allowance` raises every task
    budget by a stated amount so the run says what it measured.
    """
    outcomes: list[TaskOutcome] = []
    transport_errors = 0
    with httpx.Client(timeout=timeout) as client:
        for task in tasks:
            try:
                response = client.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions",
                    json={
                        "model": model_ref,
                        "messages": [{"role": "user", "content": task.prompt}],
                        "max_tokens": task.max_tokens + max(0, reasoning_allowance),
                        "temperature": 0,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") if isinstance(payload, dict) else None
                first = choices[0] if isinstance(choices, list) and choices else None
                message = first.get("message") if isinstance(first, dict) else None
                text = _output(message.get("content") if isinstance(message, dict) else None)
                finish = first.get("finish_reason") if isinstance(first, dict) else None
                unscorable = finish == "length" and not text.strip()
                passed = False if unscorable else bool(task.check(text))
            except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as error:
                transport_errors += 1
                text = _error(error)
                passed = False
                unscorable = False
            outcomes.append(
                TaskOutcome(
                    task.id,
                    task.category,
                    passed,
                    text[:200],
                    unscorable,
                    (
                        None
                        if unscorable or task.value_check is None
                        else bool(task.value_check(text))
                    ),
                )
            )
    if outcomes and transport_errors == len(outcomes):
        raise RuntimeError("all evaluation tasks failed at transport level")
    passed = sum(outcome.passed for outcome in outcomes)
    by_category: dict[str, float] = {}
    for category in dict.fromkeys(outcome.category for outcome in outcomes):
        category_outcomes = [outcome for outcome in outcomes if outcome.category == category]
        by_category[category] = (
            sum(outcome.passed for outcome in category_outcomes) / len(category_outcomes)
        )
    return EvalRun(
        model_ref,
        "",
        "",
        len(outcomes),
        passed,
        passed / len(outcomes) if outcomes else 0.0,
        by_category,
        outcomes,
        time.time(),
        unscorable=sum(outcome.unscorable for outcome in outcomes),
        reasoning_allowance=max(0, reasoning_allowance),
    )
