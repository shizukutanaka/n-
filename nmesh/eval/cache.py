from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from nmesh.paths import nmesh_home

from .runner import EvalRun


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
        )
    except (KeyError, TypeError, ValueError):
        return None


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
    key = f"{run.model_id}|{run.quant}|{run.backend}"
    records[key] = EvalRecord(
        run.model_id, run.quant, run.backend, run.n_tasks, run.passed,
        run.pass_rate, run.by_category, run.at,
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
