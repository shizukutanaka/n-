from __future__ import annotations

import json
from pathlib import Path

import pytest

from nmesh import cli
from nmesh.bench import BenchRecord, BenchResult, ControlledBenchResult, benchmark_key
from nmesh.catalog import ModelSpec
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _plan():
    model = ModelSpec("cli-bench", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    return build_plan(profile(64), [model], Policy(roles=["chat"]))


def _controlled(measurement: BenchResult) -> ControlledBenchResult:
    return ControlledBenchResult(
        result=measurement,
        pass_tps=(measurement.decode_tps, measurement.decode_tps),
        control_ratio=1.0,
        stable=True,
    )


def test_bench_runs_rejects_zero() -> None:
    with pytest.raises(SystemExit):
        cli.main(["bench", "--runs", "0"])


def test_bench_json_reports_spread_and_warns(monkeypatch, capsys) -> None:
    plan = _plan()
    measurement = BenchResult(
        400.0, 20.0, 0.5, False, 330, "timings", 0, 10.0, 30.0, 3,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: object())
    monkeypatch.setattr(cli, "_service_running", lambda *_args: True)
    monkeypatch.setattr(cli, "load_records", dict)
    monkeypatch.setattr(cli, "save_records", lambda _records: None)
    monkeypatch.setattr(cli, "measure_controlled", lambda *_args, **_kwargs:
                        _controlled(measurement))
    assert cli.main(["bench", "--json", "--runs", "3"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["runs"] == 3
    assert result["decode_tps_min"] == 10.0
    assert result["decode_tps_max"] == 30.0
    assert result["decode_spread"] == 1.0

    assert cli.main(["bench", "--runs", "3"]) == 0
    assert "not reproducible" in capsys.readouterr().out


def test_bench_narrow_spread_has_no_reproducibility_warning(monkeypatch, capsys) -> None:
    plan = _plan()
    measurement = BenchResult(
        400.0, 20.0, 0.5, False, 330, "timings", 0, 19.0, 21.0, 3,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: object())
    monkeypatch.setattr(cli, "_service_running", lambda *_args: True)
    monkeypatch.setattr(cli, "load_records", dict)
    monkeypatch.setattr(cli, "save_records", lambda _records: None)
    monkeypatch.setattr(cli, "measure_controlled", lambda *_args, **_kwargs:
                        _controlled(measurement))
    assert cli.main(["bench", "--runs", "3"]) == 0
    assert "not reproducible" not in capsys.readouterr().out


def test_bench_json_keeps_session_tps_when_control_is_rejected(monkeypatch, capsys) -> None:
    plan = _plan()
    service = plan.services[0]
    key = benchmark_key(
        service.model_id,
        service.quant,
        service.backend,
        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
        service.n_gpu_layers,
        service.kv_quant,
        service.spec,
    )
    previous = BenchRecord(
        48.0, 47.0, 49.0, 3, 2, 0.99, True, "old", "bench-v1",
        (48.0, 48.2),
    )
    noisy = ControlledBenchResult(
        result=BenchResult(
            100.0, 5.0, 0.5, False, 330, "timings", 0, 4.9, 5.1, 6,
        ),
        pass_tps=(5.0, 45.0),
        control_ratio=5.0 / 45.0,
        stable=False,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: object())
    monkeypatch.setattr(cli, "_service_running", lambda *_args: True)
    monkeypatch.setattr(cli, "load_records", lambda: {key: previous})
    monkeypatch.setattr(cli, "save_records", lambda _records: None)
    monkeypatch.setattr(cli, "measure_controlled", lambda *_args, **_kwargs: noisy)

    assert cli.main(["bench", "--json", "--passes", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["session_tps"] == 5.0
    assert result["median_tps"] == 48.0
    assert result["stored"] is False


def test_bench_reference_is_measured_around_controlled_passes(
    monkeypatch, capsys,
) -> None:
    plan = _plan()
    measurement = BenchResult(
        400.0, 20.0, 0.5, False, 330, "timings", 0, 19.0, 21.0, 3,
    )
    saved = {}
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: object())
    monkeypatch.setattr(cli, "_service_running", lambda *_args: True)
    monkeypatch.setattr(cli, "load_records", dict)
    monkeypatch.setattr(cli, "save_records", lambda records: saved.update(records))
    monkeypatch.setattr(
        cli, "_reference_context",
        lambda _service: (Path("llama-bench"), Path("reference.gguf"), "ref"),
    )
    monkeypatch.setattr(cli, "load_history", dict)
    monkeypatch.setattr(cli, "save_history", lambda history: None)
    reference_values = iter((60.0, 62.0))
    monkeypatch.setattr(cli, "measure_reference", lambda *_args: next(reference_values))
    monkeypatch.setattr(
        cli, "measure_controlled",
        lambda *_args, **_kwargs: _controlled(measurement),
    )

    assert cli.main(["bench", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reference_tps"] == 61.0
    assert result["reference_baseline"] is None
    assert result["reference_id"] == "ref"
    assert result["epoch"] == "unknown"
    assert saved[next(iter(saved))].reference_tps == 61.0


def test_bench_no_reference_keeps_epoch_unknown(monkeypatch, capsys) -> None:
    plan = _plan()
    measurement = BenchResult(
        400.0, 20.0, 0.5, False, 330, "timings", 0, 19.0, 21.0, 3,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: object())
    monkeypatch.setattr(cli, "_service_running", lambda *_args: True)
    monkeypatch.setattr(cli, "load_records", dict)
    monkeypatch.setattr(cli, "save_records", lambda _records: None)
    monkeypatch.setattr(
        cli, "_reference_context",
        lambda _service: (_ for _ in ()).throw(AssertionError("reference used")),
    )
    monkeypatch.setattr(
        cli, "measure_reference",
        lambda *_args: (_ for _ in ()).throw(AssertionError("reference used")),
    )
    monkeypatch.setattr(
        cli, "measure_controlled",
        lambda *_args, **_kwargs: _controlled(measurement),
    )

    assert cli.main(["bench", "--json", "--no-reference"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reference_tps"] is None
    assert result["reference_baseline"] is None
    assert result["reference_id"] == ""
    assert result["epoch"] == "unknown"
