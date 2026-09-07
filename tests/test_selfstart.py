from __future__ import annotations

from types import SimpleNamespace

from nmesh import cli
from nmesh.runtime import RuntimeStatus


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
        "profile": None,
        "_simulated": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _plan(missing_backends: list[str], runnable: bool) -> SimpleNamespace:
    return SimpleNamespace(
        missing_backends=missing_backends,
        runnable=runnable,
        services=[],
        install_hints=["Install llama.cpp with nmesh engine install"],
        policy=SimpleNamespace(),
    )


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
    plans = [_plan(["llamacpp"], False), _plan([], True)]
    plans[1].services = [{
        "service": "chat",
        "backend": "llamacpp",
        "model_ref": "model",
        "port": 18001,
        "context": 1024,
        "parallel_slots": 1,
        "n_gpu_layers": 0,
        "argv": ["llama-server"],
    }]
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

    assert cli._runtime(_up_args()) == 0
    assert install_calls == 1
    assert make_calls == 2
    assert len(runtime_calls) == 1


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
