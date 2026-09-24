from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from .depth import padded_prompt
from .suite import Task


@dataclass(frozen=True)
class TaskOutcome:
    id: str
    category: str
    passed: bool
    output: str
    unscorable: bool = False
    value_passed: bool | None = None
    failure_kind: str = ""


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
    transport_errors: int = 0
    cache_prompt: bool | None = None
    depth: int = 0
    prompt_tokens_max: int = 0


def _output(value: object) -> str:
    return value if isinstance(value, str) else ""


def _error(error: Exception) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}"[:200]


def _failure_kind(
    task: Task,
    passed: bool,
    unscorable: bool,
    value_passed: bool | None,
) -> str:
    if passed or unscorable:
        return ""
    if task.grades == "form":
        return "form"
    if task.grades == "value":
        return "value"
    return "form" if value_passed else "value"


def run(
    tasks: Iterable[Task],
    base_url: str,
    model_ref: str,
    *,
    timeout: float | None = None,
    reasoning_allowance: int = 0,
    cache_prompt: bool | None = None,
    depth: int = 0,
    on_outcome: Callable[[TaskOutcome], None] | None = None,
    parallel: int = 1,
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

    Request timeouts are derived from each task's token budget and requested
    prompt depth by default: ``30 + (max_tokens + allowance) / 2 + depth / 20``.
    Quiet prefill here measured about 242 tokens per second, while PR #58
    measured multi-minute episodes about 9x slower (about 27 tokens per
    second), so a 20-token-per-second prefill floor remains below the worst
    observed episode. This prevents the 62-second prefill at about 15k tokens
    from colliding with the 62-second budget of a 64-token task.
    """
    items = list(tasks)

    def _run_one(
        client: httpx.Client, task: Task
    ) -> tuple[TaskOutcome, int | None]:
        transport = False
        reported: int | None = None
        request_timeout = (
            timeout
            if timeout is not None
            else 30.0 + (
                task.max_tokens + max(0, reasoning_allowance)
            ) / 2.0 + depth / 20.0
        )
        try:
            prompt = (
                padded_prompt(task.prompt, depth, f"{task.id}|{depth}")
                if depth > 0 and not task.category.startswith("context.")
                else task.prompt
            )
            request: dict[str, object] = {
                "model": model_ref,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": task.max_tokens + max(0, reasoning_allowance),
                "temperature": 0,
                "stream": False,
            }
            if cache_prompt is not None:
                request["cache_prompt"] = cache_prompt
            response = client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                json=request,
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") if isinstance(payload, dict) else None
            value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                reported = value
            choices = payload.get("choices") if isinstance(payload, dict) else None
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get("message") if isinstance(first, dict) else None
            text = _output(message.get("content") if isinstance(message, dict) else None)
            finish = first.get("finish_reason") if isinstance(first, dict) else None
            unscorable = finish == "length" and not text.strip()
            passed = False if unscorable else bool(task.check(text))
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as error:
            transport = True
            text = _error(error)
            passed = False
            unscorable = False
        value_passed = (
            None
            if transport or unscorable or task.value_check is None
            else bool(task.value_check(text))
        )
        outcome = TaskOutcome(
            task.id,
            task.category,
            passed,
            text[:200],
            unscorable,
            value_passed,
            "transport"
            if transport
            else _failure_kind(task, passed, unscorable, value_passed),
        )
        return outcome, reported

    workers = max(1, int(parallel))
    results: list[tuple[TaskOutcome, int | None]] = []
    if workers == 1:
        with httpx.Client() as client:
            for task in items:
                results.append(_run_one(client, task))
    else:
        # Each worker thread owns one keep-alive client; pool.map preserves
        # task order so on_outcome progress stays sequential and deterministic.
        thread_clients = threading.local()
        owned_clients: list[httpx.Client] = []

        def _worker_client() -> httpx.Client:
            client = getattr(thread_clients, "client", None)
            if client is None:
                client = httpx.Client()
                owned_clients.append(client)
                thread_clients.client = client
            return client

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda t: _run_one(_worker_client(), t), items))
        finally:
            for client in owned_clients:
                client.close()

    outcomes: list[TaskOutcome] = []
    transport_errors = 0
    prompt_tokens_max = 0
    for outcome, reported in results:
        outcomes.append(outcome)
        if outcome.failure_kind == "transport":
            transport_errors += 1
        if reported is not None:
            prompt_tokens_max = max(prompt_tokens_max, reported)
        if on_outcome is not None:
            on_outcome(outcome)
    if outcomes and transport_errors == len(outcomes):
        raise RuntimeError("all evaluation tasks failed at transport level")
    passed_count: int = sum(outcome.passed for outcome in outcomes)
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
        passed_count,
        passed_count / len(outcomes) if outcomes else 0.0,
        by_category,
        outcomes,
        time.time(),
        unscorable=sum(outcome.unscorable for outcome in outcomes),
        reasoning_allowance=max(0, reasoning_allowance),
        transport_errors=transport_errors,
        cache_prompt=cache_prompt,
        depth=depth,
        prompt_tokens_max=prompt_tokens_max,
    )
