from __future__ import annotations

import json
from dataclasses import asdict

from nmesh.bench.cache import (
    BENCH_HARNESS_VERSION,
    BenchRecord,
    benchmark_key,
    load_cache,
    load_records,
)
from nmesh.catalog import ModelSpec
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _record(harness: str) -> BenchRecord:
    return BenchRecord(
        20.0,
        19.0,
        21.0,
        3,
        2,
        0.98,
        True,
        "now",
        harness,
        (20.0, 20.1),
    )


def test_load_cache_filters_old_harnesses_but_load_records_keeps_them(
    tmp_path,
) -> None:
    current = _record(BENCH_HARNESS_VERSION)
    payload = {
        "legacy-float": 48.54,
        "old-dict": asdict(_record("bench-v1")),
        "current": asdict(current),
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_records(path)
    assert set(records) == {"legacy-float", "old-dict", "current"}
    assert set(load_cache(path)) == {"current"}
    assert load_cache(path)["current"] == current.tps


def test_legacy_benchmark_is_estimated_and_warns_without_current_evidence() -> None:
    model = ModelSpec(
        "harness-check", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    key = benchmark_key(model.id, "q4_k_m", "llamacpp", "cpu", 0)
    plan = build_plan(
        profile(64),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
        bench_records={key: _record("legacy")},
    )

    assert plan.services
    assert plan.services[0].estimated
    assert sum("older harness" in warning for warning in plan.warnings) == 1


def test_current_harness_benchmark_does_not_warn() -> None:
    model = ModelSpec(
        "harness-current", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    key = benchmark_key(model.id, "q4_k_m", "llamacpp", "cpu", 0)
    plan = build_plan(
        profile(64),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
        bench_records={key: _record(BENCH_HARNESS_VERSION)},
    )

    assert not any("older harness" in warning for warning in plan.warnings)
