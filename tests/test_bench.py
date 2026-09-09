from __future__ import annotations

import json
from typing import ClassVar, Self

from nmesh.bench import BenchRecord, merge_measurement, runner
from nmesh.bench.runner import BenchResult, measure, measure_controlled


class _Stream:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(self.lines)


class _Client:
    lines: ClassVar[list[str]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def stream(self, _method: str, _url: str, *, json: dict[str, object]) -> _Stream:
        return _Stream(self.lines)


def _timing_lines(predicted_n: int, predicted_ms: float) -> list[str]:
    return [
        "data: " + json.dumps({
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": predicted_n,
            },
            "timings": {
                "prompt_n": 32,
                "prompt_ms": 100.0,
                "predicted_n": predicted_n,
                "predicted_ms": predicted_ms,
            },
            "choices": [],
        }),
        *(
            'data: {"choices": [{"delta": {"content": "a"}}]}'
            for _ in range(predicted_n)
        ),
        "data: [DONE]",
    ]


def _measure_once(
    monkeypatch,
    predicted_n: int,
    predicted_ms: float,
) -> BenchResult:
    _Client.lines = _timing_lines(predicted_n, predicted_ms)
    monkeypatch.setattr(runner.httpx, "Client", _Client)
    monkeypatch.setattr(
        runner.time,
        "perf_counter",
        iter([0.0, *(0.1 + index * 0.001 for index in range(predicted_n))]).__next__,
    )
    service = type("Service", (), {"model_ref": "model"})()
    return runner._measure_once(service, "http://test", 32, predicted_n)


def test_decode_rate_uses_only_decode_steps(monkeypatch) -> None:
    one = _measure_once(monkeypatch, 1, 0.001)
    two = _measure_once(monkeypatch, 2, 22.234)
    long = _measure_once(monkeypatch, 128, 2700.0)

    assert one.decode_tokens_served == 1
    assert one.decode_tps == 0.0
    assert one.decode_tps != 1_000_000.0
    assert two.decode_tokens_served == 2
    assert two.decode_tps == 1 / 0.022234
    assert long.decode_tokens_served == 128
    assert long.decode_tps == 127 / 2.7


def test_measure_and_controlled_propagate_served_token_medians(monkeypatch) -> None:
    results = iter([
        BenchResult(10.0, 20.0, 0.1, False, 32, "timings", 0, 20.0, 20.0, 1, 2),
        BenchResult(11.0, 21.0, 0.1, False, 32, "timings", 0, 21.0, 21.0, 1, 4),
        BenchResult(12.0, 22.0, 0.1, False, 32, "timings", 0, 22.0, 22.0, 1, 6),
    ])
    monkeypatch.setattr(runner, "_measure_once", lambda *_args, **_kwargs: next(results))
    measured = measure(object(), "http://test", runs=3)
    assert measured.decode_tokens_served == 4

    passes = iter([
        BenchResult(10.0, 20.0, 0.1, False, 32, "timings", 0, 20.0, 20.0, 1, 8),
        BenchResult(11.0, 21.0, 0.1, False, 32, "timings", 0, 21.0, 21.0, 1, 10),
    ])
    monkeypatch.setattr(runner, "measure", lambda *_args, **_kwargs: next(passes))
    controlled = measure_controlled(object(), "http://test", passes=2)
    assert controlled.result.decode_tokens_served == 9


def test_bench_harness_change_discards_old_sessions_and_rejections() -> None:
    records = {
        "key": BenchRecord(
            10.0, 9.0, 11.0, 3, 2, 1.0, True, "old", "bench-v1",
            (10.0, 9.0), rejected=(8.0,),
        )
    }
    record = merge_measurement(
        records,
        "key",
        tps=12.0,
        decode_tps_min=11.0,
        decode_tps_max=13.0,
        runs=3,
        passes=2,
        control_ratio=1.0,
        measured_at="new",
        harness="bench-v2",
    )
    assert record.tps == 12.0
    assert record.sessions == (12.0,)
    assert record.rejected == ()


def test_bench_same_harness_keeps_existing_merge_behavior() -> None:
    records = {
        "key": BenchRecord(
            10.0, 9.0, 11.0, 3, 2, 1.0, True, "old", "bench-v2",
            (10.0, 9.0), rejected=(8.0,),
        )
    }
    record = merge_measurement(
        records,
        "key",
        tps=12.0,
        decode_tps_min=11.0,
        decode_tps_max=13.0,
        runs=3,
        passes=2,
        control_ratio=1.0,
        measured_at="new",
        harness="bench-v2",
    )
    assert record.sessions == (12.0, 10.0, 9.0)
    assert record.rejected == (8.0,)


def test_bench_harness_change_does_not_preserve_rejected_stable_evidence() -> None:
    records = {
        "key": BenchRecord(
            10.0, 9.0, 11.0, 3, 2, 1.0, True, "old", "bench-v1",
            (10.0, 9.0), rejected=(8.0,),
        )
    }
    record = merge_measurement(
        records,
        "key",
        tps=5.0,
        decode_tps_min=4.0,
        decode_tps_max=6.0,
        runs=3,
        passes=2,
        control_ratio=0.2,
        measured_at="new",
        harness="bench-v2",
    )
    assert record.tps == 5.0
    assert record.stable is False
    assert record.sessions == (5.0,)
    assert record.rejected == (5.0,)
