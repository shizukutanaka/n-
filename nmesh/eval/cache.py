from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nmesh.paths import nmesh_home
from nmesh.persist import finite_number, int_at_least, write_results_cache

from .runner import EvalRun


@dataclass(frozen=True)
class EvalSummary:
    pass_rate: float
    passed: int
    n_tasks: int
    task_results: Mapping[str, bool]
    suite: str = ""
    digest: str = ""
    reasoning_allowance: int = 0
    cache_prompt: bool | None = None


@dataclass(frozen=True)
class EvalRecord:
    model_id: str
    quant: str
    backend: str
    n_tasks: int
    passed: int
    pass_rate: float
    by_category: dict[str, float]
    at: float
    task_results: dict[str, bool] = field(default_factory=dict)
    artifact: str = ""
    suite: str = "core"
    digest: str = ""
    unscorable: int = 0
    reasoning_allowance: int = 0
    transport_errors: int = 0
    cache_prompt: bool | None = None
    depth: int = 0
    prompt_tokens_max: int = 0


def _record(data: object) -> EvalRecord | None:
    if not isinstance(data, dict):
        return None
    try:
        model_id = data["model_id"]
        quant = data["quant"]
        backend = data["backend"]
        by_category = data["by_category"]
        if (
            not isinstance(model_id, str)
            or not isinstance(quant, str)
            or not isinstance(backend, str)
            or not isinstance(by_category, dict)
        ):
            return None
        n_tasks = data["n_tasks"]
        passed = data["passed"]
        pass_rate = data["pass_rate"]
        at = data["at"]
        task_results = data.get("task_results", {})
        artifact = data.get("artifact", "")
        suite = data.get("suite", "core")
        digest = data.get("digest", "")
        unscorable = data.get("unscorable", 0)
        allowance = data.get("reasoning_allowance", 0)
        transport_errors = data.get("transport_errors", 0)
        cache_prompt = data.get("cache_prompt")
        depth = data.get("depth", 0)
        prompt_tokens_max = data.get("prompt_tokens_max", 0)
        if (
            not int_at_least(n_tasks, 0)
            or not int_at_least(passed, 0)
            or passed > n_tasks
            or not finite_number(pass_rate)
            or not 0.0 <= pass_rate <= 1.0
            or not finite_number(at)
            or not isinstance(task_results, dict)
            or not isinstance(artifact, str)
            or not isinstance(suite, str)
            or not isinstance(digest, str)
            or not int_at_least(unscorable, 0)
            or unscorable > n_tasks
            or not int_at_least(allowance, 0)
            or not int_at_least(transport_errors, 0)
            or transport_errors > n_tasks
            or (
                cache_prompt is not None
                and not isinstance(cache_prompt, bool)
            )
            or not int_at_least(depth, 0)
            or not int_at_least(prompt_tokens_max, 0)
            or any(
                not isinstance(key, str) or not isinstance(value, bool)
                for key, value in task_results.items()
            )
        ):
            return None
        categories = {}
        for key, value in by_category.items():
            if (
                not isinstance(key, str)
                or not finite_number(value)
                or not 0.0 <= value <= 1.0
            ):
                return None
            categories[key] = float(value)
        return EvalRecord(
            model_id,
            quant,
            backend,
            n_tasks,
            passed,
            float(pass_rate),
            categories,
            float(at),
            dict(task_results),
            artifact,
            suite,
            digest,
            unscorable,
            allowance,
            transport_errors,
            cache_prompt,
            depth,
            prompt_tokens_max,
        )
    except (KeyError, TypeError, ValueError):
        return None


def eval_key(
    model_id: str,
    quant: str,
    backend: str,
    suite: str,
    digest: str,
    allowance: int = 0,
    cache_prompt: bool | None = None,
    depth: int = 0,
) -> str:
    """Identify a measurement. A token budget change is a measurement change, so
    runs made with a reasoning allowance never land on an allowance-free key.
    A depth change is also a measurement change, so deep runs get their own
    key."""
    suffix = f"|a{allowance}" if allowance else ""
    if cache_prompt is not None:
        suffix += "|c1" if cache_prompt else "|c0"
    if depth > 0:
        suffix += f"|d{depth}"
    return f"{model_id}|{quant}|{backend}|{suite}|{digest}{suffix}"


def load_eval_cache(path: Path | None = None) -> dict[str, EvalRecord]:
    target = path or (nmesh_home() / "eval.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict):
            return {}
        return {
            str(key): record
            for key, value in results.items()
            if (record := _record(value)) is not None
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_eval(run: EvalRun, path: Path | None = None) -> Path:
    target = path or (nmesh_home() / "eval.json")
    records = load_eval_cache(target)
    key = eval_key(
        run.model_id, run.quant, run.backend, run.suite, run.digest,
        run.reasoning_allowance,
        run.cache_prompt,
        run.depth,
    )
    records[key] = EvalRecord(
        model_id=run.model_id,
        quant=run.quant,
        backend=run.backend,
        n_tasks=run.n_tasks,
        passed=run.passed,
        pass_rate=run.pass_rate,
        by_category=run.by_category,
        at=run.at,
        task_results={outcome.id: outcome.passed for outcome in run.outcomes},
        artifact=run.artifact,
        suite=run.suite,
        digest=run.digest,
        unscorable=run.unscorable,
        reasoning_allowance=run.reasoning_allowance,
        transport_errors=run.transport_errors,
        cache_prompt=run.cache_prompt,
        depth=run.depth,
        prompt_tokens_max=run.prompt_tokens_max,
    )
    write_results_cache(target, records)
    return target
