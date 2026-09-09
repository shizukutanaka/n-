from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nmesh.bench.epoch import refutes
from nmesh.paths import nmesh_home

BenchCache = dict[str, float]
CACHE_PATH = nmesh_home() / "bench.json"
BENCH_HARNESS_VERSION = "bench-v2"
MIN_CONTROL_RATIO = 0.90


@dataclass(frozen=True)
class BenchRecord:
    """A reproducibility-aware benchmark record."""

    tps: float
    decode_tps_min: float
    decode_tps_max: float
    runs: int
    passes: int
    control_ratio: float | None
    stable: bool
    measured_at: str
    harness: str
    sessions: tuple[float, ...]
    rejected: tuple[float, ...] = ()
    last_control_ratio: float | None = None
    last_rejected_at: str = ""
    last_rejected_min: float | None = None
    last_rejected_max: float | None = None
    reference_tps: float | None = None
    reference_id: str = ""
    epoch: str = "unknown"
    last_rejected_reference_tps: float | None = None
    last_rejected_reference_id: str = ""
    last_rejected_epoch: str = "unknown"

    @property
    def confirmations(self) -> int:
        if not self.sessions:
            return 0
        newest = self.sessions[0]
        return sum(
            1
            for value in self.sessions
            if value == newest
            or (
                max(value, newest) > 0
                and min(value, newest) / max(value, newest) >= MIN_CONTROL_RATIO
            )
        )


def benchmark_key(model_id: str, quant: str, backend: str, gpu_name: str,
                  n_gpu_layers: int | None, kv_quant: str = "f16",
                  spec: str = "none") -> str:
    suffix = "" if kv_quant == "f16" else f"|kv{kv_quant}"
    if spec != "none":
        suffix += f"|sp{spec}"
    return f"{model_id}|{quant}|{backend}|{gpu_name}|{n_gpu_layers or 0}{suffix}"


def load_cache(path: Path | None = None) -> BenchCache:
    return {
        key: record.tps
        for key, record in load_records(path).items()
        if record.stable and record.epoch != "degraded"
    }


def _legacy(value: object) -> BenchRecord | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    tps = float(value)
    return BenchRecord(
        tps=tps,
        decode_tps_min=tps,
        decode_tps_max=tps,
        runs=1,
        passes=1,
        control_ratio=None,
        stable=True,
        measured_at="",
        harness="legacy",
        sessions=(tps,),
    )


def _record(value: object) -> BenchRecord | None:
    legacy = _legacy(value)
    if legacy is not None:
        return legacy
    if not isinstance(value, dict):
        return None
    try:
        sessions = tuple(float(item) for item in value["sessions"])
        rejected = tuple(float(item) for item in value.get("rejected", ()))
        tps = float(value["tps"])
        minimum = float(value["decode_tps_min"])
        maximum = float(value["decode_tps_max"])
        runs = int(value["runs"])
        passes = int(value["passes"])
        ratio = value.get("control_ratio")
        last_ratio = value.get("last_control_ratio")
        ratio = None if ratio is None else float(ratio)
        last_ratio = None if last_ratio is None else float(last_ratio)
        last_min = value.get("last_rejected_min")
        last_max = value.get("last_rejected_max")
        last_min = None if last_min is None else float(last_min)
        last_max = None if last_max is None else float(last_max)
        reference_value = value.get("reference_tps")
        reference_tps = (
            None if reference_value is None else float(reference_value)
        )
        rejected_reference_value = value.get("last_rejected_reference_tps")
        last_rejected_reference_tps = (
            None
            if rejected_reference_value is None
            else float(rejected_reference_value)
        )
        stable = value["stable"]
        measured_at = value["measured_at"]
        harness = value["harness"]
        reference_id = str(value.get("reference_id", ""))
        epoch = str(value.get("epoch", "unknown"))
        last_rejected_reference_id = str(
            value.get("last_rejected_reference_id", "")
        )
        last_rejected_epoch = str(value.get("last_rejected_epoch", "unknown"))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        isinstance(stable, bool) is False
        or not isinstance(measured_at, str)
        or not isinstance(harness, str)
        or runs < 1
        or passes < 1
        or (not sessions and epoch != "degraded")
    ):
        return None
    return BenchRecord(
        tps=tps,
        decode_tps_min=minimum,
        decode_tps_max=maximum,
        runs=runs,
        passes=passes,
        control_ratio=ratio,
        stable=stable,
        measured_at=measured_at,
        harness=harness,
        sessions=sessions[:5],
        rejected=rejected[:3],
        last_control_ratio=last_ratio,
        last_rejected_at=str(value.get("last_rejected_at", "")),
        last_rejected_min=last_min,
        last_rejected_max=last_max,
        reference_tps=reference_tps,
        reference_id=reference_id,
        epoch=epoch,
        last_rejected_reference_tps=last_rejected_reference_tps,
        last_rejected_reference_id=last_rejected_reference_id,
        last_rejected_epoch=last_rejected_epoch,
    )


def load_records(path: Path | None = None) -> dict[str, BenchRecord]:
    target = path or CACHE_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    records: dict[str, BenchRecord] = {}
    for key, value in payload.items():
        record = _record(value)
        if record is not None:
            records[str(key)] = record
    return records


def save_cache(cache: BenchCache, path: Path | None = None) -> Path:
    target = path or CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return target


def save_records(records: dict[str, BenchRecord], path: Path | None = None) -> Path:
    target = path or CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({key: asdict(value) for key, value in records.items()}, indent=2),
        encoding="utf-8",
    )
    return target


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def demote_stale(
    records: dict[str, BenchRecord],
    reference_id: str,
    reference_tps: float,
) -> tuple[str, ...]:
    """Invalidate evidence disproved by a later, faster reference."""
    demoted: list[str] = []
    timestamp = _now()
    for key, previous in records.items():
        if (
            not reference_id
            or previous.reference_id != reference_id
            or previous.reference_tps is None
            or previous.reference_tps <= 0
            or not refutes(previous.reference_tps, reference_tps)
        ):
            continue
        records[key] = BenchRecord(
            tps=previous.tps,
            decode_tps_min=previous.decode_tps_min,
            decode_tps_max=previous.decode_tps_max,
            runs=previous.runs,
            passes=previous.passes,
            control_ratio=previous.control_ratio,
            stable=False,
            measured_at=previous.measured_at,
            harness=previous.harness,
            sessions=(),
            rejected=(previous.tps, *previous.rejected)[:3],
            last_control_ratio=previous.control_ratio,
            last_rejected_at=timestamp,
            last_rejected_min=previous.decode_tps_min,
            last_rejected_max=previous.decode_tps_max,
            reference_tps=previous.reference_tps,
            reference_id=previous.reference_id,
            epoch="degraded",
            last_rejected_reference_tps=previous.reference_tps,
            last_rejected_reference_id=previous.reference_id,
            last_rejected_epoch=previous.epoch,
        )
        demoted.append(key)
    return tuple(sorted(demoted))


def merge_measurement(
    records: dict[str, BenchRecord],
    key: str,
    *,
    tps: float,
    decode_tps_min: float,
    decode_tps_max: float,
    runs: int,
    passes: int,
    control_ratio: float | None,
    measured_at: str | None = None,
    harness: str = BENCH_HARNESS_VERSION,
    reference_tps: float | None = None,
    reference_id: str = "",
    epoch: str = "unknown",
) -> BenchRecord:
    """Merge one controlled measurement while retaining usable evidence."""
    timestamp = measured_at or _now()
    stable = (
        passes >= 2
        and control_ratio is not None
        and control_ratio >= MIN_CONTROL_RATIO
        and epoch != "degraded"
    )
    previous = records.get(key)
    comparable = previous is not None and previous.harness == harness
    if stable:
        sessions = (
            (tps, *previous.sessions)[:5]
            if comparable and previous is not None and previous.stable
            else (tps,)
        )
        newest = sessions[0]
        agreeing = tuple(
            value for value in sessions
            if value == newest
            or (
                max(value, newest) > 0
                and min(value, newest) / max(value, newest) >= MIN_CONTROL_RATIO
            )
        )
        record = BenchRecord(
            tps=statistics.median(agreeing),
            decode_tps_min=decode_tps_min,
            decode_tps_max=decode_tps_max,
            runs=runs,
            passes=passes,
            control_ratio=control_ratio,
            stable=True,
            measured_at=timestamp,
            harness=harness,
            sessions=sessions,
            rejected=previous.rejected if comparable and previous is not None else (),
            last_control_ratio=(
                previous.last_control_ratio
                if comparable and previous is not None else None
            ),
            last_rejected_at=(
                previous.last_rejected_at
                if comparable and previous is not None else ""
            ),
            last_rejected_min=(
                previous.last_rejected_min
                if comparable and previous is not None else None
            ),
            last_rejected_max=(
                previous.last_rejected_max
                if comparable and previous is not None else None
            ),
            reference_tps=reference_tps,
            reference_id=reference_id,
            epoch=epoch,
            last_rejected_reference_tps=(
                previous.last_rejected_reference_tps
                if comparable and previous is not None else None
            ),
            last_rejected_reference_id=(
                previous.last_rejected_reference_id
                if comparable and previous is not None else ""
            ),
            last_rejected_epoch=(
                previous.last_rejected_epoch
                if comparable and previous is not None else "unknown"
            ),
        )
    elif comparable and previous is not None and previous.stable:
        record = BenchRecord(
            tps=previous.tps,
            decode_tps_min=previous.decode_tps_min,
            decode_tps_max=previous.decode_tps_max,
            runs=previous.runs,
            passes=previous.passes,
            control_ratio=previous.control_ratio,
            stable=True,
            measured_at=previous.measured_at,
            harness=previous.harness,
            sessions=previous.sessions,
            rejected=(tps, *previous.rejected)[:3],
            last_control_ratio=control_ratio,
            last_rejected_at=timestamp,
            last_rejected_min=decode_tps_min,
            last_rejected_max=decode_tps_max,
            reference_tps=previous.reference_tps,
            reference_id=previous.reference_id,
            epoch=previous.epoch,
            last_rejected_reference_tps=reference_tps,
            last_rejected_reference_id=reference_id,
            last_rejected_epoch=epoch,
        )
    else:
        rejected = (
            (tps, *previous.rejected)[:3]
            if comparable and previous is not None else (tps,)
        )
        record = BenchRecord(
            tps=tps,
            decode_tps_min=decode_tps_min,
            decode_tps_max=decode_tps_max,
            runs=runs,
            passes=passes,
            control_ratio=control_ratio,
            stable=False,
            measured_at=timestamp,
            harness=harness,
            sessions=(tps,),
            rejected=rejected,
            last_control_ratio=control_ratio,
            last_rejected_at=timestamp,
            last_rejected_min=decode_tps_min,
            last_rejected_max=decode_tps_max,
            reference_tps=reference_tps,
            reference_id=reference_id,
            epoch=epoch,
            last_rejected_reference_tps=reference_tps,
            last_rejected_reference_id=reference_id,
            last_rejected_epoch=epoch,
        )
    records[key] = record
    return record


def benchmark(run: Callable[[int, int], float], prefill_tokens: int = 512,
              decode_tokens: int = 128, runs: int = 3) -> float:
    values = [run(prefill_tokens, decode_tokens) for _ in range(runs)]
    return statistics.median(values)


def autotune(run: Callable[[int, int], float], contexts: list[int],
             gpu_layers: list[int]) -> tuple[int, int, float]:
    best = (contexts[0], gpu_layers[0], float("-inf"))
    for context in contexts:
        for layers in gpu_layers:
            value = run(context, layers)
            if value > best[2]:
                best = (context, layers, value)
    return best
