from __future__ import annotations

from pathlib import Path

from nmesh.bench import (
    EPOCH_MIN_RATIO,
    BenchRecord,
    demote_stale,
    load_cache,
    load_records,
    save_records,
)
from nmesh.bench.epoch import (
    EPOCH_HISTORY,
    EpochSample,
    baseline,
    choose_reference_model,
    classify,
    find_reference_binary,
    load_history,
    prune_degraded,
    refutes,
    save_history,
)


def _samples(reference_id: str, values: list[float]) -> tuple[EpochSample, ...]:
    return tuple(
        EpochSample(reference_id, value, f"2025-01-01T00:00:{index:02d}+00:00")
        for index, value in enumerate(values)
    )


def test_baseline_uses_upper_median_and_empty_is_unknown() -> None:
    history = {"ref": _samples("ref", [10.0, 20.0, 30.0, 40.0])}
    assert baseline(history, "ref") == 35.0
    assert baseline(history, "missing") is None


def test_classify_accepts_the_ratio_boundary() -> None:
    assert classify(80.0, 100.0) == "healthy"
    assert classify(79.9, 100.0) == "degraded"
    assert classify(100.0, None) == "unknown"


def test_refutes_uses_the_epoch_threshold_and_rejects_invalid_values() -> None:
    assert not refutes(0.0, 50.0)
    assert not refutes(float("nan"), 50.0)
    assert not refutes(40.0, 0.0)
    assert not refutes(40.0, 40.0 / EPOCH_MIN_RATIO - 0.01)
    assert refutes(40.0, 40.0 / EPOCH_MIN_RATIO)
    assert refutes(40.0, 40.0 / EPOCH_MIN_RATIO + 0.01)


def test_prune_degraded_drops_slow_samples_and_keeps_boundary() -> None:
    current = 43.2
    samples = _samples("ref", [19.0, 41.0, current * EPOCH_MIN_RATIO])
    retained = prune_degraded(samples, current)
    assert [sample.tps for sample in retained] == [41.0, current * EPOCH_MIN_RATIO]


def test_history_is_trimmed_and_reference_ids_are_isolated(tmp_path) -> None:
    path = tmp_path / "epoch.json"
    history = {
        "one": _samples("one", [float(value) for value in range(1, 21)]),
        "two": _samples("two", [80.0, 81.0, 82.0, 83.0]),
    }
    save_history(history, path)
    loaded = load_history(path)
    assert len(loaded["one"]) == EPOCH_HISTORY
    assert loaded["one"][0].tps == 1.0
    assert loaded["two"] == history["two"]
    assert baseline(loaded, "two") == 82.5


def test_reference_discovery_uses_smallest_model_and_windows_sibling(tmp_path) -> None:
    server = tmp_path / "llama-server.exe"
    binary = tmp_path / "llama-bench.exe"
    server.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")
    (tmp_path / "large.gguf").write_bytes(b"1234")
    (tmp_path / "small.gguf").write_bytes(b"12")
    assert find_reference_binary(server) == binary
    assert choose_reference_model(tmp_path) == tmp_path / "small.gguf"


def test_measure_reference_reads_llama_bench_json_without_shelling_out(monkeypatch) -> None:
    from nmesh.bench import epoch

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": '[{"avg_ts": 61.5}]', "stderr": ""},
        )()

    monkeypatch.setattr(epoch.subprocess, "run", fake_run)
    assert epoch.measure_reference(
        Path("llama-bench"), Path("reference.gguf"),
    ) == 61.5
    assert calls[0][0] == [
        "llama-bench", "-m", "reference.gguf", "-p", "0", "-n", "32",
        "-r", "2", "-t", "8", "-o", "json",
    ]


def test_measure_reference_timeout_scales_with_model_size(monkeypatch, tmp_path) -> None:
    from nmesh.bench import epoch

    seen_timeouts: list[float] = []

    def fake_run(command, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": '[{"avg_ts": 42.0}]', "stderr": ""},
        )()

    monkeypatch.setattr(epoch.subprocess, "run", fake_run)

    small = tmp_path / "small.gguf"
    small.write_bytes(b"\0" * 1024)
    assert epoch.measure_reference(Path("llama-bench"), small) == 42.0
    # Small model: flat floor applies (300s, decode floor is below it).
    assert seen_timeouts[-1] == 300.0

    big = tmp_path / "big.gguf"
    # Sparse file: st_size reports 40 GiB without allocating the bytes.
    with big.open("wb") as handle:
        handle.truncate(40 * 1024**3)
    epoch.measure_reference(Path("llama-bench"), big)
    # 40 GiB at a 50 MiB/s floor: ~858s of cold-load bound + decode floor.
    assert seen_timeouts[-1] > 800.0
    assert seen_timeouts[-1] <= 3600.0


def test_measure_reference_timeout_budgets_all_split_shards(
    monkeypatch, tmp_path,
) -> None:
    from nmesh.bench import epoch

    seen_timeouts: list[float] = []

    def fake_run(command, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": '[{"avg_ts": 42.0}]', "stderr": ""},
        )()

    monkeypatch.setattr(epoch.subprocess, "run", fake_run)

    for part in range(1, 5):
        shard = tmp_path / f"model-0000{part}-of-00004.gguf"
        with shard.open("wb") as handle:
            handle.truncate(10 * 1024**3)
    epoch.measure_reference(
        Path("llama-bench"), tmp_path / "model-00001-of-00004.gguf",
    )
    # llama-bench reads all four 10 GiB shards — 40 GiB total — even though
    # only part 1 is named on the command line.
    assert seen_timeouts[-1] > 800.0
    assert seen_timeouts[-1] <= 3600.0


def _record(reference_id: str, reference_tps: float | None) -> BenchRecord:
    return BenchRecord(
        tps=20.0,
        decode_tps_min=19.0,
        decode_tps_max=21.0,
        runs=3,
        passes=2,
        control_ratio=1.0,
        stable=True,
        measured_at="2025-01-01T00:00:00+00:00",
        harness="bench-v1",
        sessions=(20.0, 20.0),
        reference_tps=reference_tps,
        reference_id=reference_id,
        epoch="healthy",
    )


def test_demote_stale_invalidates_faster_disproved_reference(tmp_path) -> None:
    records = {"stale": _record("ref", 20.0)}
    assert 48.0 / 20.0 > 1 / EPOCH_MIN_RATIO
    assert demote_stale(records, "ref", 48.0) == ("stale",)
    demoted = records["stale"]
    assert demoted.epoch == "degraded"
    assert demoted.stable is False
    assert demoted.sessions == ()
    assert demoted.tps == 20.0
    assert demoted.rejected[0] == 20.0
    assert demoted.last_rejected_reference_tps == 20.0
    assert demoted.last_rejected_epoch == "healthy"
    path = tmp_path / "bench.json"
    save_records(records, path)
    assert load_records(path)["stale"].sessions == ()
    assert load_cache(path) == {}


def test_demote_stale_accepts_exact_epoch_boundary() -> None:
    records = {"boundary": _record("ref", 20.0)}
    assert demote_stale(records, "ref", 20.0 / EPOCH_MIN_RATIO) == (
        "boundary",
    )


def test_demote_stale_ignores_matching_noise_and_unrelated_records() -> None:
    records = {
        "noise": _record("ref", 45.0),
        "other": _record("other", 20.0),
        "empty": _record("", 20.0),
        "missing": _record("ref", None),
    }
    assert 48.0 / 45.0 <= 1 / EPOCH_MIN_RATIO
    assert demote_stale(records, "ref", 48.0) == ()
    assert all(record.epoch == "healthy" for record in records.values())


def test_demote_stale_requires_reference_identity_and_value() -> None:
    records = {
        "other": _record("other", 20.0),
        "empty": _record("", 20.0),
        "missing": _record("ref", None),
    }
    assert demote_stale(records, "ref", 48.0) == ()
