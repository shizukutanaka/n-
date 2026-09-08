from __future__ import annotations

import json
import math
import os
import statistics
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from nmesh.paths import nmesh_home

REFERENCE_PREFILL_TOKENS = 512
"""Reference depth used by ``nmesh.bench.runner.measure()``."""

COMPARABLE_DEPTH_FACTOR = 4.0
"""Maximum comparable depth based on the measured 512-token noise band.

Depths through 2048 stayed within this host's 6.4% healthy noise of the
512-token reference, while 4096 and deeper prompts fell materially below it.
"""

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
    prefill_tps: float | None = None
    in_flight: int | None = None
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class OverlayReport:
    values: dict[str, float]
    under_load: int
    off_reference: int
    unknown_depth: int


class Telemetry:
    def __init__(self, path: Path | None = None,
                 max_samples: int = 200) -> None:
        self.path = path or nmesh_home() / "telemetry.json"
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
                float(item["prefill_tps"]) if item.get("prefill_tps") is not None else None,
                int(item["in_flight"]) if item.get("in_flight") is not None else None,
                int(item["prompt_tokens"]) if item.get("prompt_tokens") is not None else None,
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

    @staticmethod
    def _summarize(values: list[Sample]) -> dict[str, float]:
        item: dict[str, float] = {"samples": float(len(values))}
        decode = [value.decode_tps for value in values if value.decode_tps is not None]
        prefill = [value.prefill_tps for value in values if value.prefill_tps is not None]
        ttft = [value.ttft_s for value in values if value.ttft_s is not None]
        total = [value.total_s for value in values]
        if decode:
            item["decode_tps_median"] = statistics.median(decode)
        if prefill:
            item["prefill_tps_median"] = statistics.median(prefill)
        if ttft:
            ordered = sorted(ttft)
            item["ttft_s_median"] = statistics.median(ordered)
            item["ttft_s_p95"] = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        if total:
            item["total_s_median"] = statistics.median(total)
        return item

    def summary(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[Sample]] = {}
        for sample in self.samples():
            groups.setdefault(sample.service, []).append(sample)
        return {
            service: self._summarize(values)
            for service, values in groups.items()
        }

    def summary_by_approximate(self) -> dict[str, dict[bool, dict[str, float]]]:
        groups: dict[str, dict[bool, list[Sample]]] = {}
        for sample in self.samples():
            groups.setdefault(sample.service, {}).setdefault(sample.approximate, []).append(sample)
        return {
            service: {
                approximate: self._summarize(values)
                for approximate, values in grouped.items()
            }
            for service, grouped in groups.items()
        }

    def bench_overlay(self, min_samples: int = 5) -> dict[str, float]:
        """Return the single-stream decode overlay.

        One CPU machine measured Qwen2.5-1.5B-Instruct-Q4_K_M at 46.20 tok/s
        alone with N=4, 46.26 tok/s alone with N=8, 36.64 tok/s with 4 requests
        in flight, and 23.30 tok/s with 8 requests in flight, using one
        llama.cpp server, ``-t 8``, ``-np N``, ``-c 4096*N``, ``/completions``,
        ``n_predict=128``, ``temperature=0``, ``top_k=1``, ``cache_prompt=false``,
        and nonce-prefixed prompts. Aggregate server throughput rose while the
        per-request rate fell. This is one model on one CPU machine with
        llama.cpp and does not generalize.
        """
        return self.overlay_report(min_samples).values

    def overlay_report(self, min_samples: int = 5) -> OverlayReport:
        exact: dict[str, list[float]] = {}
        approximate: dict[str, list[float]] = {}
        under_load = 0
        off_reference = 0
        unknown_depth = 0
        for sample in self.samples():
            if sample.decode_tps is not None:
                if sample.in_flight != 1:
                    under_load += 1
                    continue
                if sample.prompt_tokens is None:
                    unknown_depth += 1
                    continue
                if sample.prompt_tokens > REFERENCE_PREFILL_TOKENS * COMPARABLE_DEPTH_FACTOR:
                    off_reference += 1
                    continue
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
        return OverlayReport(
            values={
                key: statistics.median(values)
                for key, values in selected.items() if len(values) >= min_samples
            },
            under_load=under_load,
            off_reference=off_reference,
            unknown_depth=unknown_depth,
        )


_default = Telemetry()


def record(sample: Sample) -> None:
    _default.record(sample)


def summary() -> dict[str, dict[str, float]]:
    return _default.summary()


def summary_by_approximate() -> dict[str, dict[bool, dict[str, float]]]:
    return _default.summary_by_approximate()


def bench_overlay(min_samples: int = 5) -> dict[str, float]:
    return _default.bench_overlay(min_samples)


def overlay_report(min_samples: int = 5) -> OverlayReport:
    return _default.overlay_report(min_samples)


__all__ = [
    "COMPARABLE_DEPTH_FACTOR",
    "REFERENCE_PREFILL_TOKENS",
    "OverlayReport",
    "Sample",
    "Telemetry",
    "bench_overlay",
    "overlay_report",
    "record",
    "summary",
    "summary_by_approximate",
]
