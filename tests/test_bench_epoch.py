from __future__ import annotations

from pathlib import Path

from nmesh.bench.epoch import (
    EPOCH_HISTORY,
    EpochSample,
    baseline,
    choose_reference_model,
    classify,
    find_reference_binary,
    load_history,
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
