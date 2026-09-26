"""Use a shared reference workload as a validity gate, never a correction factor.

The reference throughput is kept separate from candidate throughput: candidate
measurements remain observed values and are never rescaled.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nmesh import evidence
from nmesh.paths import nmesh_home

EPOCH_MIN_RATIO = evidence.EPOCH_MIN_RATIO
refutes = evidence.refutes
EPOCH_HISTORY = 12
EPOCH_PATH = nmesh_home() / "epoch.json"

#: llama-bench must read the whole GGUF before generating; a flat timeout
#: kills the reference workload on slow storage (the same cold-load bound the
#: supervisor's health wait already scales for). 50 MiB/s is an HDD-class
#: read floor, matching HEALTH_LOAD_FLOOR_BPS in runtime/supervisor.py.
REFERENCE_LOAD_FLOOR_BPS = 50.0 * 1024 * 1024
REFERENCE_TIMEOUT_MIN = 300.0
REFERENCE_TIMEOUT_MAX = 3600.0


@dataclass(frozen=True)
class EpochSample:
    reference_id: str
    tps: float
    measured_at: str


def find_reference_binary(server: Path) -> Path | None:
    """Find the llama-bench executable next to llama-server."""
    candidates = (
        server.with_name("llama-bench.exe"),
        server.with_name("llama-bench"),
    )
    return next((item for item in candidates if item.is_file()), None)


def choose_reference_model(models_dir: Path) -> Path | None:
    """Choose the smallest local GGUF with a deterministic tie-break."""
    try:
        candidates = [
            item for item in models_dir.glob("*.gguf")
            if item.is_file()
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item.stat().st_size, item.name.casefold(), item.name))
    except OSError:
        return None


def reference_id(engine_build: str, model: Path, threads: int, gen: int) -> str:
    return f"{engine_build}|{model.name}|{model.stat().st_size}|t{threads}|n{gen}"


def _reference_timeout(model: Path, gen: int, reps: int) -> float:
    """Bound scaled to the model's cold read plus a slow decode floor.

    A flat bound worked only when the reference GGUF loaded in far less than
    300s; on slow storage a healthy measurement was killed mid-load and the
    epoch check silently degraded to ``unknown`` on exactly the large models
    that need it most.
    """
    try:
        size = model.stat().st_size
    except OSError:
        size = 0
    load_bound = size / REFERENCE_LOAD_FLOOR_BPS
    decode_bound = max(gen, 0) * max(reps, 1) / 2.0  # 2 tok/s decode floor
    return min(
        max(REFERENCE_TIMEOUT_MIN, load_bound + decode_bound),
        REFERENCE_TIMEOUT_MAX,
    )


def measure_reference(
    binary: Path,
    model: Path,
    threads: int = 8,
    gen: int = 32,
    reps: int = 2,
) -> float:
    command = [
        str(binary),
        "-m", str(model),
        "-p", "0",
        "-n", str(gen),
        "-r", str(reps),
        "-t", str(threads),
        "-o", "json",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_reference_timeout(model, gen, reps),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"reference workload failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"reference workload failed: {detail}")
    try:
        rows = json.loads(result.stdout)
        value = float(rows[0]["avg_ts"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("reference workload returned invalid JSON") from error
    if value <= 0:
        raise RuntimeError("reference workload returned non-positive throughput")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history(path: Path | None = None) -> dict[str, tuple[EpochSample, ...]]:
    target = path or EPOCH_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    history: dict[str, tuple[EpochSample, ...]] = {}
    for key, values in payload.items():
        if not isinstance(values, list):
            continue
        samples: list[EpochSample] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            try:
                sample = EpochSample(
                    reference_id=str(value["reference_id"]),
                    tps=float(value["tps"]),
                    measured_at=str(value["measured_at"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if sample.reference_id != str(key) or sample.tps <= 0:
                continue
            samples.append(sample)
        if samples:
            history[str(key)] = tuple(samples[:EPOCH_HISTORY])
    return history


def save_history(
    history: dict[str, tuple[EpochSample, ...]],
    path: Path | None = None,
) -> Path:
    target = path or EPOCH_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: [asdict(sample) for sample in samples[:EPOCH_HISTORY]]
        for key, samples in history.items()
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def baseline(
    history: dict[str, tuple[EpochSample, ...]],
    reference_id: str,
) -> float | None:
    values = sorted(sample.tps for sample in history.get(reference_id, ()))
    if not values:
        return None
    return statistics.median(values[len(values) // 2:])


def classify(current: float, base: float | None) -> str:
    if base is None:
        return "unknown"
    return "healthy" if current >= base * EPOCH_MIN_RATIO else "degraded"


def prune_degraded(
    samples: tuple[EpochSample, ...],
    current: float,
) -> tuple[EpochSample, ...]:
    """Drop samples disproved by a later, faster reference."""
    return tuple(
        sample
        for sample in samples
        if classify(sample.tps, current) != "degraded"
    )
