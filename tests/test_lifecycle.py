from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import psutil

from nmesh import cli
from nmesh.planner import Policy, build_plan
from nmesh.runtime.service_unit import service_unit
from nmesh.runtime.supervisor import Supervisor

from .test_planner import profile


def test_gateway_state_is_terminated_and_removed(tmp_path: Path) -> None:
    terminated: list[int] = []
    supervisor = Supervisor(
        state_path=tmp_path / "state.json",
        terminator=terminated.append,
    )
    supervisor.record_gateway(os.getpid(), 18000)

    assert supervisor.down(foreign=True).running is False
    assert terminated == [os.getpid()]
    assert not (tmp_path / "state.json").exists()


def test_gateway_non_owner_is_retained(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "gateway": {
                    "pid": os.getpid(),
                    "create_time": psutil.Process(os.getpid()).create_time(),
                    "port": 18000,
                    "owner_pid": os.getpid() + 1,
                },
                "services": [],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    supervisor.down()

    assert terminated == []
    assert state_path.exists()
    assert "gateway" in json.loads(state_path.read_text(encoding="utf-8"))


def test_gateway_creation_mismatch_is_pruned_without_termination(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "gateway": {
                    "pid": os.getpid(),
                    "create_time": psutil.Process(os.getpid()).create_time() + 100,
                    "port": 18000,
                    "owner_pid": os.getpid() + 1,
                },
                "services": [],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    supervisor.down(foreign=True)

    assert terminated == []
    assert not state_path.exists()


def test_status_surfaces_gateway_pid_and_port(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    supervisor = Supervisor(state_path=state_path)
    supervisor.record_gateway(os.getpid(), 18001)

    status = supervisor.status()
    gateway = next(item for item in status.services if item["service"] == "gateway")

    assert gateway["pid"] == os.getpid()
    assert gateway["port"] == 18001
    assert gateway["running"] is True


def test_live_foreign_service_keeps_state_when_gateway_is_stopped(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "gateway": {"pid": 999999, "port": 18000, "owner_pid": os.getpid() + 1},
                "services": [{
                    "service": "chat",
                    "pid": os.getpid(),
                    "create_time": psutil.Process(os.getpid()).create_time(),
                    "port": 18010,
                    "owner_pid": os.getpid() + 1,
                }],
            }
        ),
        encoding="utf-8",
    )
    supervisor = Supervisor(state_path=state_path, terminator=lambda _pid: None)

    supervisor.down()

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(payload["services"]) == 1
    assert "gateway" not in payload


def test_launch_gateway_uses_windows_detachment(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pid = 1234

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli, "record_gateway", lambda _pid, _port: None)
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    cli._launch_gateway(19000, detach=True)

    kwargs = calls[0][1]
    assert kwargs["creationflags"] == 0x00000008 | 0x00000200
    assert kwargs["close_fds"] is True
    assert kwargs["stdout"] is kwargs["stderr"]


def test_launch_gateway_uses_posix_detachment(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeProcess:
        pid = 1234

    def popen(_command, **kwargs):
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli, "record_gateway", lambda _pid, _port: None)
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    cli._launch_gateway(19000, detach=True)

    assert calls[0]["start_new_session"] is True
    assert calls[0]["close_fds"] is True


def test_fallback_only_scales_context_with_parallel_flag(tmp_path: Path) -> None:
    from nmesh.catalog import load_catalog

    plan = build_plan(profile(32, (24,)), load_catalog(), Policy(roles=["chat"]))
    service = plan.services[0]
    launch = service.launch
    supervisor = Supervisor(state_path=tmp_path / "state.json")
    for layer_flag in ("-ngl", "--gpu-layers", "--n-gpu-layers"):
        service = replace(
            service,
            memory=replace(service.memory, parallel_slots=3),
            n_gpu_layers=7,
            launch=replace(
                launch,
                argv=["llama-server", "-c", "1", "--parallel", "3", layer_flag, "4", "--other", "x"],
                health_url=None,
            ),
        )
        scaled = supervisor._fallback(replace(plan, services=[service]), 2).services[0]
        assert scaled.launch.argv[scaled.launch.argv.index("-c") + 1] == str(scaled.context * 3)
        assert scaled.launch.argv[scaled.launch.argv.index(layer_flag) + 1] == str(scaled.n_gpu_layers)

    plain_service = replace(
        service,
        launch=replace(launch, argv=["llama-server", "-c", "1", "--other", "x"], health_url=None),
    )
    plain = supervisor._fallback(replace(plan, services=[plain_service]), 2).services[0]
    assert plain.launch.argv[plain.launch.argv.index("-c") + 1] == str(plain.context)


def test_service_units_are_pure_and_platform_specific(monkeypatch) -> None:
    monkeypatch.setattr("nmesh.runtime.service_unit.sys.executable", "/resolved/python")

    filename, text, command = service_unit(19000, "posix")
    assert filename.endswith(".service")
    assert "/resolved/python" in text
    assert "19000" in text
    assert command.startswith("systemctl --user")

    filename, text, command = service_unit(19001, "darwin")
    assert filename.endswith(".plist")
    assert "/resolved/python" in text
    assert "19001" in text
    assert command.startswith("launchctl")

    filename, text, command = service_unit(19002, "nt")
    assert filename.endswith(".cmd")
    assert "schtasks" in text
    assert "19002" in command
