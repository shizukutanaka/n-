from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nmesh.paths import nmesh_home

from .runner import EvalRun


@dataclass(frozen=True)
class EvalSummary:
    pass_rate: float
    passed: int
    n_tasks: int
    task_results: Mapping[str, bool]


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
        if (
            isinstance(n_tasks, bool)
            or not isinstance(n_tasks, int)
            or n_tasks < 0
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or passed < 0
            or passed > n_tasks
            or isinstance(pass_rate, bool)
            or not isinstance(pass_rate, (int, float))
            or not math.isfinite(pass_rate)
            or not 0.0 <= pass_rate <= 1.0
            or isinstance(at, bool)
            or not isinstance(at, (int, float))
            or not math.isfinite(at)
            or not isinstance(task_results, dict)
            or not isinstance(artifact, str)
            or not isinstance(suite, str)
            or not isinstance(digest, str)
            or isinstance(unscorable, bool)
            or not isinstance(unscorable, int)
            or not 0 <= unscorable <= n_tasks
            or isinstance(allowance, bool)
            or not isinstance(allowance, int)
            or allowance < 0
            or isinstance(transport_errors, bool)
            or not isinstance(transport_errors, int)
            or not 0 <= transport_errors <= n_tasks
            or (
                cache_prompt is not None
                and not isinstance(cache_prompt, bool)
            )
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
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
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
) -> str:
    """Identify a measurement. A token budget change is a measurement change, so
    runs made with a reasoning allowance never land on an allowance-free key."""
    suffix = f"|a{allowance}" if allowance else ""
    if cache_prompt is not None:
        suffix += "|c1" if cache_prompt else "|c0"
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
    )
    records[key] = EvalRecord(
        run.model_id, run.quant, run.backend, run.n_tasks, run.passed,
        run.pass_rate, run.by_category, run.at,
        {outcome.id: outcome.passed for outcome in run.outcomes},
        run.artifact,
        run.suite,
        run.digest,
        run.unscorable,
        run.reasoning_allowance,
        run.transport_errors,
        run.cache_prompt,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps({"results": {key: asdict(value) for key, value in records.items()}},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target
