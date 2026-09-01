from __future__ import annotations

import json
import math
import os
import statistics
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    service: str
    key: str
    decode_tps: float | None
    ttft_s: float | None
    total_s: float
    completion_tokens: int
    at: float
    approximate: bool = True


class Telemetry:
    def __init__(self, path: Path = Path.home() / ".nmesh" / "telemetry.json",  # noqa: B008
                 max_samples: int = 200) -> None:
        self.path = path
        self.max_samples = max_samples
        self._lock = threading.Lock()

    def _load(self) -> list[Sample]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            values = payload.get("samples", []) if isinstance(payload, dict) else []
            return [Sample(
                str(item["service"]), str(item["key"]),
                float(item["decode_tps"]) if item.get("decode_tps") is not None else None,
                float(item["ttft_s"]) if item.get("ttft_s") is not None else None,
                float(item["total_s"]), int(item["completion_tokens"]), float(item["at"]),
                bool(item.get("approximate", True)),
            ) for item in values if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []

    def record(self, sample: Sample) -> None:
        with self._lock:
            samples = self._load()
            samples.append(sample)
            by_key: dict[str, list[Sample]] = {}
            for item in samples:
                by_key.setdefault(item.key, []).append(item)
            samples = [item for values in by_key.values() for item in values[-self.max_samples:]]
            payload = json.dumps({"samples": [asdict(item) for item in samples]},
                                 indent=2, ensure_ascii=False)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, self.path)
            except OSError:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def samples(self) -> list[Sample]:
        with self._lock:
            return self._load()

    def summary(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[Sample]] = {}
        for sample in self.samples():
            groups.setdefault(sample.service, []).append(sample)
        result: dict[str, dict[str, float]] = {}
        for service, values in groups.items():
            item: dict[str, float] = {"samples": float(len(values))}
            decode = [value.decode_tps for value in values if value.decode_tps is not None]
            ttft = [value.ttft_s for value in values if value.ttft_s is not None]
            total = [value.total_s for value in values]
            if decode:
                item["decode_tps_median"] = statistics.median(decode)
            if ttft:
                ordered = sorted(ttft)
                item["ttft_s_median"] = statistics.median(ordered)
                item["ttft_s_p95"] = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
            if total:
                item["total_s_median"] = statistics.median(total)
            result[service] = item
        return result

    def bench_overlay(self, min_samples: int = 5) -> dict[str, float]:
        exact: dict[str, list[float]] = {}
        approximate: dict[str, list[float]] = {}
        for sample in self.samples():
            if sample.decode_tps is not None:
                groups = approximate if sample.approximate else exact
                groups.setdefault(sample.key, []).append(sample.decode_tps)
        selected: dict[str, list[float]] = {}
        for key, values in exact.items():
            if len(values) >= min_samples or key not in approximate:
                selected[key] = values
            else:
                selected[key] = approximate[key]
        for key, values in approximate.items():
            selected.setdefault(key, values)
        return {
            key: statistics.median(values)
            for key, values in selected.items() if len(values) >= min_samples
        }


_default = Telemetry()


def record(sample: Sample) -> None:
    _default.record(sample)


def summary() -> dict[str, dict[str, float]]:
    return _default.summary()


def bench_overlay(min_samples: int = 5) -> dict[str, float]:
    return _default.bench_overlay(min_samples)


__all__ = ["Sample", "Telemetry", "bench_overlay", "record", "summary"]
