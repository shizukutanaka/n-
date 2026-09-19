from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from nmesh import cli
from nmesh.planner import (
    LAUNCH_REVISION,
    build_plan,
    load_plan,
    save_plan,
)
from nmesh.runtime import RuntimeStatus
from nmesh.telemetry import OverlayReport


def _up_args(**overrides: object) -> SimpleNamespace:
    values = {
        "command": "up",
        "port": 18000,
        "detach": False,
        "dry_run": True,
        "no_download": False,
        "ignore_free_memory": False,
        "json": False,
        "lang": None,
        "model": None,
        "ignore_eval_evidence": False,
        "kv_quant": "f16",
        "profile": None,
        "_simulated": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kv_quant_cli_reaches_policy(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(cli, "detect_hardware", lambda: object())
    monkeypatch.setattr(cli, "load_catalog", list)
    monkeypatch.setattr(cli, "overlay_report", lambda: OverlayReport({}, 0, 0, 0))
    monkeypatch.setattr(cli, "_eval_rates", dict)
    monkeypatch.setattr(
        cli,
        "build_plan",
        lambda _profile, _catalog, policy, *_args, **_kwargs:
        captured.append(policy) or object(),
    )
    cli._make_plan(SimpleNamespace(
        profile=None,
        roles="chat",
        prefer="balanced",
        context=None,
        budget="total",
        kv_quant="q8_0",
        parallel_slots=None,
        lang=None,
        model=None,
        ignore_eval_evidence=False,
    ))
    assert captured[0].kv_quant == "q8_0"


def test_invalid_kv_quant_cli_value_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["plan", "--kv-quant", "int4"])
    assert error.value.code == 2


def _plan(missing_backends: list[str], runnable: bool) -> SimpleNamespace:
    return SimpleNamespace(
        missing_backends=missing_backends,
        runnable=runnable,
        services=[],
        install_hints=["Install llama.cpp with nmesh engine install"],
        policy=SimpleNamespace(lang="en", budget_source="total"),
        tier=SimpleNamespace(value="balanced"),
        warnings=[],
    )


def _runnable_plan() -> SimpleNamespace:
    result = _plan([], True)
    result.services = [SimpleNamespace(
        name="chat",
        roles=["chat"],
        model_id="qwen2.5-1.5b-instruct",
        backend="llamacpp",
        context=4096,
        memory=SimpleNamespace(parallel_slots=1),
        n_gpu_layers=0,
        languages=["en"],
        decode_tps=12.0,
    )]
    return result


def test_plan_reports_install_hint(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_make_plan",
        lambda _args: _plan(["llamacpp"], False),
    )
    args = SimpleNamespace(profile=None, json=False, explain=False)

    assert cli._plan(args) == 1
    captured = capsys.readouterr()
    assert "plan produced no runnable services" in captured.err
    assert "nmesh engine install" in captured.err


def test_up_autoinstalls_once_rebuilds_and_reaches_runtime(monkeypatch) -> None:
    plans = [_plan(["llamacpp"], False), _runnable_plan()]
    make_calls = 0
    install_calls = 0
    runtime_calls: list[object] = []

    def make_plan(_args):
        nonlocal make_calls
        make_calls += 1
        return plans[make_calls - 1]

    def install(**_kwargs):
        nonlocal install_calls
        install_calls += 1
        return SimpleNamespace(tag="b123", variant="cpu"), []

    def runtime_up(plan, **_kwargs):
        runtime_calls.append(plan)
        return RuntimeStatus(False, [])

    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "_make_plan", make_plan)
    monkeypatch.setattr(cli.engine_runtime, "install", install)
    monkeypatch.setattr(cli, "save_plan", lambda _plan: None)
    monkeypatch.setattr(cli, "runtime_up", runtime_up)
    monkeypatch.setattr(
        cli,
        "_launch_gateway",
        lambda _port, detach: (SimpleNamespace(pid=1234), None),
    )
    monkeypatch.setattr(cli, "_wait_gateway", lambda _port, _process: True)
    monkeypatch.setattr(cli, "clear_gateway", lambda _pid: None)
    monkeypatch.setattr(cli, "disarm_atexit", lambda: None)

    assert cli._runtime(_up_args(dry_run=False, detach=True)) == 0
    assert install_calls == 1
    assert make_calls == 2
    assert len(runtime_calls) == 1


def test_up_renders_fresh_plan_but_json_suppresses_it(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        cli, "_console", lambda: Console(file=output, force_terminal=False, width=200)
    )
    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "save_plan", lambda _plan: None)
    monkeypatch.setattr(cli, "runtime_up", lambda *_args, **_kwargs: RuntimeStatus(False, []))

    monkeypatch.setattr(cli, "_make_plan", lambda _args: _runnable_plan())
    assert cli._runtime(_up_args()) == 0
    assert "qwen2.5-1.5b-instruct" in output.getvalue()

    output.seek(0)
    output.truncate(0)
    assert cli._runtime(_up_args(json=True)) == 0
    assert "qwen2.5-1.5b-instruct" not in output.getvalue()


def test_up_plan_failure_uses_plan_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(
        cli, "_make_plan", lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert cli._ensure_runnable_plan(_up_args()) is None
    captured = capsys.readouterr()
    assert "plan failed: boom" in captured.err
    assert "up failed: boom" not in captured.err


def test_up_no_download_does_not_autoinstall(monkeypatch, capsys) -> None:
    install_calls = 0

    def install(**_kwargs):
        nonlocal install_calls
        install_calls += 1
        raise AssertionError("engine install should not be called")

    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _plan(["llamacpp"], False))
    monkeypatch.setattr(cli.engine_runtime, "install", install)

    assert cli._ensure_runnable_plan(_up_args(no_download=True)) is None
    assert install_calls == 0
    assert "nmesh engine install" in capsys.readouterr().err


def test_up_dry_run_does_not_autoinstall(monkeypatch, capsys) -> None:
    install_calls = 0

    def install(**_kwargs):
        nonlocal install_calls
        install_calls += 1
        raise AssertionError("must not download in dry-run")

    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _plan(["llamacpp"], False))
    monkeypatch.setattr(cli.engine_runtime, "install", install)

    assert cli._runtime(_up_args()) == 1
    assert install_calls == 0
    captured = capsys.readouterr()
    assert "plan produced no runnable services" in captured.err
    assert "nmesh engine install" in captured.err


def test_up_does_not_autoinstall_other_missing_backend(monkeypatch) -> None:
    install_calls = 0

    def install(**_kwargs):
        nonlocal install_calls
        install_calls += 1
        raise AssertionError("engine install should not be called")

    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _plan(["ollama"], False))
    monkeypatch.setattr(cli.engine_runtime, "install", install)

    assert cli._ensure_runnable_plan(_up_args()) is None
    assert install_calls == 0


def test_up_does_not_autoinstall_simulated_profile(monkeypatch) -> None:
    install_calls = 0

    def install(**_kwargs):
        nonlocal install_calls
        install_calls += 1
        raise AssertionError("engine install should not be called")

    monkeypatch.setattr(cli, "load_plan", lambda: None)
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _plan(["llamacpp"], False))
    monkeypatch.setattr(cli.engine_runtime, "install", install)

    assert cli._ensure_runnable_plan(_up_args(profile="profile.json")) is None
    assert install_calls == 0


def test_ignored_up_plan_flags_lists_nondefaults() -> None:
    assert cli._ignored_up_plan_flags(_up_args()) == []
    assert cli._ignored_up_plan_flags(_up_args(context_shift=True)) == [
        "--context-shift"
    ]
    assert cli._ignored_up_plan_flags(
        _up_args(kv_quant="q8_0", cache_reuse=256)
    ) == ["--kv-quant", "--cache-reuse"]


def test_saved_plan_warns_when_up_flags_ignored(monkeypatch, capsys) -> None:
    saved = SimpleNamespace(
        services=[object()], runnable=True,
        launch_revision=LAUNCH_REVISION,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: saved)
    assert cli._ensure_runnable_plan(_up_args(context_shift=True)) is saved
    assert "--context-shift" in capsys.readouterr().err


def test_saved_plan_warns_when_launch_revision_stale(monkeypatch, capsys) -> None:
    saved = SimpleNamespace(
        services=[object()], runnable=True, launch_revision=0,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: saved)
    assert cli._ensure_runnable_plan(_up_args()) is saved
    err = capsys.readouterr().err
    assert "older nmesh" in err
    assert "nmesh plan" in err


def test_saved_plan_at_current_revision_stays_silent(monkeypatch, capsys) -> None:
    saved = SimpleNamespace(
        services=[object()], runnable=True,
        launch_revision=LAUNCH_REVISION,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: saved)
    assert cli._ensure_runnable_plan(_up_args()) is saved
    assert "older nmesh" not in capsys.readouterr().err


def test_plan_roundtrip_preserves_launch_revision(
    tmp_path, monkeypatch,
) -> None:
    from tests.test_planner import load_catalog, profile

    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    plan = build_plan(profile(64, (24,)), load_catalog())
    save_plan(plan)
    loaded = load_plan()
    assert loaded is not None and loaded.launch_revision == LAUNCH_REVISION
    raw = json.loads((tmp_path / "plan.json").read_text())
    assert raw["launch_revision"] == LAUNCH_REVISION
    del raw["launch_revision"]
    (tmp_path / "plan.json").write_text(json.dumps(raw))
    legacy = load_plan()
    assert legacy is not None and legacy.launch_revision == 0
