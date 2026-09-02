from __future__ import annotations

import json

import pytest

from nmesh import cli
from nmesh.bench import BenchResult
from nmesh.catalog import ModelSpec
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _plan():
    model = ModelSpec("cli-bench", "test", 500_000_000, 24, 16, 2, 64, 1024,
                      4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"})
    return build_plan(profile(64), [model], Policy(roles=["chat"]))


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
    monkeypatch.setattr(cli, "load_cache", dict)
    monkeypatch.setattr(cli, "save_cache", lambda _cache: None)
    monkeypatch.setattr(cli, "measure", lambda *_args, **_kwargs: measurement)
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
    monkeypatch.setattr(cli, "load_cache", dict)
    monkeypatch.setattr(cli, "save_cache", lambda _cache: None)
    monkeypatch.setattr(cli, "measure", lambda *_args, **_kwargs: measurement)
    assert cli.main(["bench", "--runs", "3"]) == 0
    assert "not reproducible" not in capsys.readouterr().out
