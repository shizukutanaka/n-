from __future__ import annotations

import json

import nmesh.planner.core as planner_core
from nmesh.bench import benchmark_key
from nmesh.bench.cache import (
    BenchRecord,
    load_cache,
    load_records,
    merge_measurement,
    save_records,
)
from nmesh.bench.runner import BenchResult, measure_controlled
from nmesh.catalog import ModelSpec
from nmesh.planner import Policy

from .test_planner import profile


def _result(tps: float) -> BenchResult:
    return BenchResult(
        prefill_tps=tps * 2,
        decode_tps=tps,
        ttft_s=0.1,
        approximate=False,
        prompt_tokens=8,
        prefill_source="timings",
        cached_prompt_tokens=0,
        decode_tps_min=tps - 1,
        decode_tps_max=tps + 1,
        runs=3,
    )


def test_bench_records_round_trip_and_legacy(tmp_path) -> None:
    path = tmp_path / "bench.json"
    record = BenchRecord(
        20.0, 19.0, 21.0, 3, 2, 0.98, True, "now", "bench-v1",
        (20.0, 20.5),
    )
    save_records({"record": record}, path)
    loaded = load_records(path)
    assert loaded["record"] == record

    path.write_text(json.dumps({"legacy": 48.54}), encoding="utf-8")
    loaded = load_records(path)
    assert loaded["legacy"].harness == "legacy"
    assert loaded["legacy"].confirmations == 1
    assert load_cache(path) == {"legacy": 48.54}


def test_unstable_records_are_hidden_but_prior_evidence_survives(tmp_path) -> None:
    path = tmp_path / "bench.json"
    records = {
        "key": BenchRecord(
            48.0, 47.0, 49.0, 3, 2, 0.99, True, "now", "bench-v1",
            (48.0, 48.2),
        )
    }
    merge_measurement(
        records,
        "key",
        tps=5.2,
        decode_tps_min=5.0,
        decode_tps_max=5.4,
        runs=3,
        passes=2,
        control_ratio=0.2,
    )
    save_records(records, path)
    loaded = load_records(path)["key"]
    assert loaded.stable is True
    assert loaded.tps == 48.0
    assert loaded.rejected == (5.2,)
    assert load_cache(path) == {"key": 48.0}


def test_measure_controlled_uses_pass_control(monkeypatch) -> None:
    results = iter([_result(40.0), _result(41.0)])
    monkeypatch.setattr(
        "nmesh.bench.runner.measure",
        lambda *_args, **_kwargs: next(results),
    )
    controlled = measure_controlled(object(), "http://test", passes=2)
    assert controlled.pass_tps == (40.0, 41.0)
    assert controlled.control_ratio == 40.0 / 41.0
    assert controlled.stable is True
    assert controlled.result.decode_tps == 40.5


def test_measure_controlled_rejects_unstable_and_single_pass(monkeypatch) -> None:
    results = iter([_result(5.0), _result(45.0)])
    monkeypatch.setattr(
        "nmesh.bench.runner.measure",
        lambda *_args, **_kwargs: next(results),
    )
    unstable = measure_controlled(object(), "http://test", passes=2)
    assert unstable.stable is False
    assert unstable.control_ratio < 0.90

    monkeypatch.setattr(
        "nmesh.bench.runner.measure",
        lambda *_args, **_kwargs: _result(20.0),
    )
    single = measure_controlled(object(), "http://test", passes=1)
    assert single.control_ratio is None
    assert single.stable is False


def test_planner_requires_two_agreeing_sessions_to_exclude() -> None:
    model = ModelSpec(
        "bench-confirmation", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    policy = Policy(roles=["chat"])
    key = benchmark_key(model.id, "q4_k_m", "llamacpp", "cpu", 0)
    cache = {key: 5.05}
    unconfirmed = BenchRecord(
        5.05, 5.0, 5.1, 3, 2, 0.95, True, "now", "bench-v1", (5.05,),
    )
    candidates = planner_core._candidate_for(
        model, profile(64), policy, cache, bench_records={key: unconfirmed},
    )
    assert any(item.quant == "q4_k_m" for item in candidates)

    confirmed = BenchRecord(
        5.05, 5.0, 5.1, 3, 2, 0.95, True, "now", "bench-v1",
        (5.05, 5.1),
    )
    candidates = planner_core._candidate_for(
        model, profile(64), policy, cache, bench_records={key: confirmed},
    )
    assert not any(item.quant == "q4_k_m" for item in candidates)


def test_unstable_record_isolated_and_stable_measurement_restores_evidence(
    tmp_path,
) -> None:
    records = {
        "unstable": BenchRecord(
            5.0, 4.9, 5.1, 3, 2, 0.1, False, "old", "bench-v1", (5.0,),
        ),
        "other": BenchRecord(
            20.0, 19.0, 21.0, 3, 2, 0.99, True, "old", "bench-v1",
            (20.0, 20.2),
        ),
    }
    path = tmp_path / "bench.json"
    save_records(records, path)
    assert load_cache(path) == {"other": 20.0}

    merge_measurement(
        records,
        "unstable",
        tps=6.0,
        decode_tps_min=5.9,
        decode_tps_max=6.1,
        runs=3,
        passes=2,
        control_ratio=0.99,
    )
    save_records(records, path)
    loaded = load_records(path)
    assert loaded["unstable"].stable is True
    assert loaded["unstable"].tps == 6.0
    assert loaded["unstable"].confirmations == 1
    assert load_cache(path) == {"unstable": 6.0, "other": 20.0}
