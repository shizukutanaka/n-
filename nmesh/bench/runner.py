from __future__ import annotations

import json
import statistics
import time
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


_FILLER = "benchmark filler text "


def _prompt(tokens: int) -> str:
    """Create a prompt using approximately four characters per token."""
    return _FILLER * max(1, round(tokens * 4 / len(_FILLER)))


def _measure_once(service: PlannedService, base_url: str, prefill_tokens: int,
                  decode_tokens: int) -> BenchResult:
    body = {
        "model": service.model_ref,
        "messages": [{"role": "user", "content": _prompt(prefill_tokens)}],
        "max_tokens": decode_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first_time: float | None = None
    last_time: float | None = None
    chunks = 0
    usage: dict[str, object] | None = None
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
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
                continue
            now = time.perf_counter()
            chunks += 1
            first_time = now if first_time is None else first_time
            last_time = now
    if first_time is None or last_time is None:
        raise RuntimeError("upstream returned no SSE chunks")
    ttft = max(first_time - started, 0.000001)
    elapsed = max(last_time - first_time, 0.000001)
    if usage is not None and {
        "prompt_tokens", "completion_tokens"
    } <= usage.keys():
        prompt_count = int(usage["prompt_tokens"])
        completion_count = int(usage["completion_tokens"])
        return BenchResult(
            prompt_count / ttft,
            max(completion_count - 1, 0) / elapsed,
            ttft,
            False,
            prompt_count,
        )
    return BenchResult(
        prefill_tokens / ttft,
        max(chunks - 1, 0) / elapsed,
        ttft,
        True,
        None,
    )


def measure(service: PlannedService, base_url: str, prefill_tokens: int = 512,
            decode_tokens: int = 128, runs: int = 3) -> BenchResult:
    results = [_measure_once(service, base_url, prefill_tokens, decode_tokens) for _ in range(runs)]
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
    )
