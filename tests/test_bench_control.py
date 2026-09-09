from __future__ import annotations

import json
from typing import ClassVar

import nmesh.planner.core as planner_core
from nmesh.bench import benchmark_key, runner
from nmesh.bench.cache import (
    BENCH_HARNESS_VERSION,
    BenchRecord,
    load_cache,
    load_records,
    merge_measurement,
    save_records,
)
from nmesh.bench.runner import BenchResult, measure, measure_controlled
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
    assert loaded["legacy"].epoch == "unknown"
    assert loaded["legacy"].confirmations == 1
    assert load_cache(path) == {}


def test_unstable_records_are_hidden_but_prior_evidence_survives(tmp_path) -> None:
    path = tmp_path / "bench.json"
    records = {
        "key": BenchRecord(
            48.0, 47.0, 49.0, 3, 2, 0.99, True, "now", BENCH_HARNESS_VERSION,
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
        harness=BENCH_HARNESS_VERSION,
    )
    save_records(records, path)
    loaded = load_records(path)["key"]
    assert loaded.stable is True
    assert loaded.tps == 48.0
    assert loaded.rejected == (5.2,)
    assert load_cache(path) == {"key": 48.0}


def test_degraded_epoch_preserves_prior_record_and_is_hidden_without_session(
    tmp_path,
) -> None:
    records = {
        "key": BenchRecord(
            48.0, 47.0, 49.0, 3, 2, 0.99, True, "old", BENCH_HARNESS_VERSION,
            (48.0, 48.2), reference_tps=61.0, reference_id="ref", epoch="healthy",
        )
    }
    merge_measurement(
        records,
        "key",
        tps=21.0,
        decode_tps_min=20.0,
        decode_tps_max=22.0,
        runs=3,
        passes=2,
        control_ratio=0.99,
        reference_tps=24.0,
        reference_id="ref",
        epoch="degraded",
        harness=BENCH_HARNESS_VERSION,
    )
    record = records["key"]
    assert record.tps == 48.0
    assert record.sessions == (48.0, 48.2)
    assert record.confirmations == 2
    assert record.rejected == (21.0,)
    assert record.last_rejected_reference_tps == 24.0
    assert record.last_rejected_reference_id == "ref"
    path = tmp_path / "bench.json"
    save_records(records, path)
    assert load_cache(path) == {"key": 48.0}


def test_epoch_fields_round_trip_and_degraded_cache_filter(tmp_path) -> None:
    path = tmp_path / "bench.json"
    records = {
        "healthy": BenchRecord(
            40.0, 39.0, 41.0, 3, 2, 0.99, True, "now", BENCH_HARNESS_VERSION,
            (40.0,), reference_tps=60.0, reference_id="ref", epoch="healthy",
        ),
        "degraded": BenchRecord(
            20.0, 19.0, 21.0, 3, 2, 0.99, True, "now", BENCH_HARNESS_VERSION,
            (20.0,), reference_tps=24.0, reference_id="ref", epoch="degraded",
        ),
    }
    save_records(records, path)
    loaded = load_records(path)
    assert loaded == records
    assert load_cache(path) == {"healthy": 40.0}


def test_degraded_epoch_without_prior_evidence_is_not_cached(tmp_path) -> None:
    records: dict[str, BenchRecord] = {}
    merge_measurement(
        records,
        "new",
        tps=21.0,
        decode_tps_min=20.0,
        decode_tps_max=22.0,
        runs=3,
        passes=2,
        control_ratio=0.99,
        reference_tps=24.0,
        reference_id="ref",
        epoch="degraded",
    )
    path = tmp_path / "bench.json"
    save_records(records, path)
    loaded = load_records(path)["new"]
    assert loaded.stable is False
    assert loaded.sessions == (21.0,)
    assert load_cache(path) == {}


def test_unknown_epoch_keeps_the_existing_merge_behavior() -> None:
    implicit: dict[str, BenchRecord] = {}
    explicit: dict[str, BenchRecord] = {}
    values = {
        "tps": 40.0,
        "decode_tps_min": 39.0,
        "decode_tps_max": 41.0,
        "runs": 3,
        "passes": 2,
        "control_ratio": 0.99,
        "measured_at": "fixed",
    }
    implicit_record = merge_measurement(implicit, "key", **values)
    explicit_record = merge_measurement(explicit, "key", **values, epoch="unknown")
    assert implicit_record == explicit_record
    assert implicit_record.epoch == "unknown"


def test_measure_controlled_uses_pass_control(monkeypatch) -> None:
    results = iter([_result(40.0), _result(41.0)])
    cache_prompts = []
    monkeypatch.setattr(
        "nmesh.bench.runner.measure",
        lambda *_args, **kwargs: (
            cache_prompts.append(kwargs["cache_prompt"]) or next(results)
        ),
    )
    controlled = measure_controlled(
        object(), "http://test", passes=2, cache_prompt=False,
    )
    assert controlled.pass_tps == (40.0, 41.0)
    assert controlled.control_ratio == 40.0 / 41.0
    assert controlled.stable is True
    assert controlled.result.decode_tps == 40.5
    assert cache_prompts == [False, False]


def test_measure_forwards_cache_prompt(monkeypatch) -> None:
    cache_prompts = []
    monkeypatch.setattr(
        "nmesh.bench.runner._measure_once",
        lambda *_args, **kwargs: (
            cache_prompts.append(kwargs["cache_prompt"]) or _result(40.0)
        ),
    )
    measure(object(), "http://test", runs=2, cache_prompt=True)
    assert cache_prompts == [True, True]


class _BenchStream:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(self._lines)


class _BenchClient:
    requests: ClassVar[list[dict[str, object]]] = []
    lines: ClassVar[list[str]] = []

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def stream(self, _method, _url, *, json):
        self.requests.append(json)
        return _BenchStream(self.lines)


def _ttft_lines(prompt_tokens: int, cached_tokens: int) -> list[str]:
    return [
        (
            "data: "
            + json.dumps({
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                },
                "choices": [],
            })
        ),
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]


def test_bench_request_cache_prompt_and_ttft_cached_token_fallback(monkeypatch) -> None:
    service = type("Service", (), {"model_ref": "model"})()
    _BenchClient.requests = []
    _BenchClient.lines = _ttft_lines(100, 60)
    monkeypatch.setattr(runner.httpx, "Client", _BenchClient)
    monkeypatch.setattr(runner.time, "perf_counter", iter((0.0, 2.0)).__next__)
    uncached = runner._measure_once(service, "http://test", 100, 8)
    assert "cache_prompt" not in _BenchClient.requests[-1]
    assert uncached.prefill_tps == 40 / 2
    assert uncached.prefill_source == "ttft"

    _BenchClient.lines = _ttft_lines(70, 60)
    monkeypatch.setattr(runner.time, "perf_counter", iter((0.0, 2.0)).__next__)
    cached = runner._measure_once(
        service, "http://test", 100, 8, cache_prompt=False,
    )
    assert _BenchClient.requests[-1]["cache_prompt"] is False
    assert cached.prefill_tps == 10 / 2
    assert cached.prefill_source == "cached"

    _BenchClient.lines = _ttft_lines(100, 0)
    monkeypatch.setattr(runner.time, "perf_counter", iter((0.0, 2.0)).__next__)
    no_cache = runner._measure_once(
        service, "http://test", 100, 8, cache_prompt=True,
    )
    assert _BenchClient.requests[-1]["cache_prompt"] is True
    assert no_cache.prefill_tps == 100 / 2
    assert no_cache.prefill_source == "ttft"


def test_bench_prompt_nonce_leads_filler() -> None:
    assert runner._prompt(32, "nonce").startswith("nonce ")
    # The leading nonce keeps prompt reuse from contaminating measurements.


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
        5.05, 5.0, 5.1, 3, 2, 0.95, True, "now", BENCH_HARNESS_VERSION, (5.05,),
    )
    candidates = planner_core._candidate_for(
        model, profile(64), policy, cache, bench_records={key: unconfirmed},
    )
    assert any(item.quant == "q4_k_m" for item in candidates)

    confirmed = BenchRecord(
        5.05, 5.0, 5.1, 3, 2, 0.95, True, "now", BENCH_HARNESS_VERSION,
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
            5.0, 4.9, 5.1, 3, 2, 0.1, False, "old", BENCH_HARNESS_VERSION, (5.0,),
        ),
        "other": BenchRecord(
            20.0, 19.0, 21.0, 3, 2, 0.99, True, "old", BENCH_HARNESS_VERSION,
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
