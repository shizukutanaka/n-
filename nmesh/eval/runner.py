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
) -> EvalRun:
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
                        "max_tokens": task.max_tokens,
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
                passed = bool(task.check(text))
            except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as error:
                transport_errors += 1
                text = _error(error)
                passed = False
            outcomes.append(TaskOutcome(task.id, task.category, passed, text[:200]))
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
    )
