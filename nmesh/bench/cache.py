from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path

from nmesh.paths import nmesh_home

BenchCache = dict[str, float]
CACHE_PATH = nmesh_home() / "bench.json"


def benchmark_key(model_id: str, quant: str, backend: str, gpu_name: str,
                  n_gpu_layers: int | None) -> str:
    return f"{model_id}|{quant}|{backend}|{gpu_name}|{n_gpu_layers or 0}"


def load_cache(path: Path | None = None) -> BenchCache:
    target = path or CACHE_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return {str(key): float(value) for key, value in payload.items()} if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_cache(cache: BenchCache, path: Path | None = None) -> Path:
    target = path or CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return target


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
