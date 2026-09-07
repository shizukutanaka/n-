from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass

import httpx

from nmesh.planner import PlannedService


@dataclass(frozen=True)
class BenchResult:
    prefill_tps: float
    decode_tps: float
    ttft_s: float
    approximate: bool = True
    prompt_tokens: int | None = None
    prefill_source: str = "ttft"
    cached_prompt_tokens: int = 0
    decode_tps_min: float = 0.0
    decode_tps_max: float = 0.0
    runs: int = 1


@dataclass(frozen=True)
class ControlledBenchResult:
    """Benchmark output with an across-pass reproducibility control."""

    result: BenchResult
    pass_tps: tuple[float, ...]
    control_ratio: float | None
    stable: bool

    @property
    def measurement(self) -> BenchResult:
        return self.result


_FILLER = "benchmark filler text "


def _prompt(tokens: int, nonce: str) -> str:
    """Create a prompt using approximately four characters per token."""
    return f"{nonce} " + _FILLER * max(1, round(tokens * 4 / len(_FILLER)))


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _measure_once(service: PlannedService, base_url: str, prefill_tokens: int,
                  decode_tokens: int) -> BenchResult:
    body = {
        "model": service.model_ref,
        "messages": [{
            "role": "user",
            "content": _prompt(prefill_tokens, uuid.uuid4().hex),
        }],
        "max_tokens": decode_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first_time: float | None = None
    last_time: float | None = None
    chunks = 0
    usage: dict[str, object] | None = None
    timings: dict[str, object] | None = None
    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client, \
            client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                payload = {}
            candidate_usage = payload.get("usage") if isinstance(payload, dict) else None
            candidate_timings = payload.get("timings") if isinstance(payload, dict) else None
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
            if isinstance(candidate_timings, dict):
                timings = candidate_timings
            if (
                (isinstance(candidate_usage, dict) or isinstance(candidate_timings, dict))
                and (
                    not isinstance(choices, list)
                    or not choices
                )
            ):
                continue
            now = time.perf_counter()
            chunks += 1
            first_time = now if first_time is None else first_time
            last_time = now
    if first_time is None or last_time is None:
        raise RuntimeError("upstream returned no SSE chunks")
    ttft = max(first_time - started, 0.000001)
    elapsed = max(last_time - first_time, 0.000001)
    prompt_count = _int_value(usage.get("prompt_tokens")) if usage is not None else None
    completion_count = (
        _int_value(usage.get("completion_tokens")) if usage is not None else None
    )
    details = usage.get("prompt_tokens_details") if usage is not None else None
    cached = (
        _int_value(details.get("cached_tokens"))
        if isinstance(details, dict) else None
    )
    if cached is None:
        cached = _int_value(timings.get("cache_n")) if timings is not None else None
    cached = max(cached or 0, 0)
    prompt_n = _int_value(timings.get("prompt_n")) if timings is not None else None
    prompt_ms = _float_value(timings.get("prompt_ms")) if timings is not None else None
    predicted_n = _int_value(timings.get("predicted_n")) if timings is not None else None
    predicted_ms = (
        _float_value(timings.get("predicted_ms")) if timings is not None else None
    )
    if prompt_count is None and prompt_n is not None:
        prompt_count = prompt_n
    if prompt_n is not None and prompt_n > 0 and prompt_ms is not None and prompt_ms > 0:
        prefill_tps = prompt_n / (prompt_ms / 1000)
        # Partial cache use still leaves prompt_n/prompt_ms as exact processed work.
        prefill_source = (
            "timings" if prompt_n >= 16 and cached < prompt_n else "cached"
        )
    else:
        prefill_count = prompt_count if prompt_count is not None else prefill_tokens
        prefill_tps = prefill_count / ttft
        prefill_source = "ttft"
    if predicted_n is not None and predicted_n >= 1 and predicted_ms is not None and predicted_ms > 0:
        decode_tps = predicted_n / (predicted_ms / 1000)
    elif completion_count is not None:
        decode_tps = max(completion_count - 1, 0) / elapsed
    else:
        decode_tps = max(chunks - 1, 0) / elapsed
    approximate = not (
        prompt_count is not None and completion_count is not None
        or prompt_n is not None
        or predicted_n is not None
    )
    return BenchResult(
        prefill_tps,
        decode_tps,
        ttft,
        approximate,
        prompt_count,
        prefill_source,
        cached,
        decode_tps,
        decode_tps,
        1,
    )


def measure(service: PlannedService, base_url: str, prefill_tokens: int = 512,
            decode_tokens: int = 128, runs: int = 3) -> BenchResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    results = [_measure_once(service, base_url, prefill_tokens, decode_tokens) for _ in range(runs)]
    sources = {item.prefill_source for item in results}
    source = (
        "ttft" if "ttft" in sources
        else "cached" if "cached" in sources
        else "timings"
    )
    return BenchResult(
        statistics.median(item.prefill_tps for item in results),
        statistics.median(item.decode_tps for item in results),
        statistics.median(item.ttft_s for item in results),
        any(item.approximate for item in results),
        (
            int(statistics.median(
                item.prompt_tokens for item in results
                if item.prompt_tokens is not None
            ))
            if any(item.prompt_tokens is not None for item in results) else None
        ),
        source,
        max(item.cached_prompt_tokens for item in results),
        min(item.decode_tps for item in results),
        max(item.decode_tps for item in results),
        len(results),
    )


def measure_controlled(
    service: PlannedService,
    base_url: str,
    prefill_tokens: int = 512,
    decode_tokens: int = 128,
    runs: int = 3,
    passes: int = 2,
) -> ControlledBenchResult:
    if passes < 1:
        raise ValueError("passes must be at least 1")
    results = [
        measure(service, base_url, prefill_tokens, decode_tokens, runs)
        for _ in range(passes)
    ]
    pass_tps = tuple(item.decode_tps for item in results)
    ratio = (
        min(pass_tps) / max(pass_tps)
        if passes >= 2 and max(pass_tps) > 0
        else 0.0 if passes >= 2 else None
    )
    sources = {item.prefill_source for item in results}
    source = (
        "ttft" if "ttft" in sources
        else "cached" if "cached" in sources
        else "timings"
    )
    merged = BenchResult(
        statistics.median(item.prefill_tps for item in results),
        statistics.median(pass_tps),
        statistics.median(item.ttft_s for item in results),
        any(item.approximate for item in results),
        (
            int(statistics.median(
                item.prompt_tokens for item in results
                if item.prompt_tokens is not None
            ))
            if any(item.prompt_tokens is not None for item in results) else None
        ),
        source,
        max(item.cached_prompt_tokens for item in results),
        min(item.decode_tps_min for item in results),
        max(item.decode_tps_max for item in results),
        sum(item.runs for item in results),
    )
    return ControlledBenchResult(
        result=merged,
        pass_tps=pass_tps,
        control_ratio=ratio,
        stable=ratio is not None and ratio >= 0.90,
    )
