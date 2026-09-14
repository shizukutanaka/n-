from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import psutil

from nmesh import cli
from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan
from nmesh.runtime import service_unit as service_unit_module
from nmesh.runtime.logs import log_path, open_log, tail
from nmesh.runtime.service_unit import launcher_script, service_unit
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


def test_runtime_log_rotation_and_tail(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setenv("NMESH_LOG_MAX_BYTES", "4")
    first = open_log("chat")
    first.write(b"old\n")
    first.close()
    second = open_log("chat")
    second.write(b"new\n")
    second.close()

    assert log_path("chat").read_bytes() == b"new\n"
    assert log_path("chat").with_name("chat.log.1").read_bytes() == b"old\n"
    assert tail("chat") == ["new"]


def test_runtime_log_tail_handles_missing_and_invalid_utf8(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    assert tail("missing") == []
    path = log_path("chat")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"one\nbad \xff\n\nthree\n")

    assert tail("chat", 2) == ["bad \ufffd", "three"]


def test_supervisor_captures_backend_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    plan = build_plan(profile(64, (24,)), load_catalog(), Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        launch=replace(
            plan.services[0].launch,
            argv=[sys.executable, "-c", "print('captured backend output')"],
            health_url=None,
        ),
    )
    supervisor = Supervisor()
    process = supervisor._launch(service)
    assert process.wait(timeout=10) == 0

    assert "captured backend output" in log_path(service.name).read_text()


def test_unhealthy_message_includes_backend_log_tail(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    handle = open_log("chat")
    handle.write(b"backend failed to start\n")
    handle.close()

    message = Supervisor()._unhealthy_message("chat")

    assert "Service did not become healthy: chat" in message
    assert "backend failed to start" in message


def test_logs_cli(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    handle = open_log("chat")
    handle.write(b"first\nsecond\n")
    handle.close()

    assert cli.main(["logs"]) == 0
    assert capsys.readouterr().out.strip() == "chat"
    assert cli.main(["logs", "chat", "--lines", "1"]) == 0
    assert capsys.readouterr().out.strip() == "second"
    assert cli.main(["logs", "chat", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["service"] == "chat"
    assert payload["lines"] == ["first", "second"]
    assert cli.main(["logs", "missing"]) == 1
    assert "No log found" in capsys.readouterr().err
    assert cli.main(["logs", "../x"]) == 1
    assert "No log found" in capsys.readouterr().err


def test_unload_cli(monkeypatch, capsys) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"unloaded": ["chat"]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert cli.main(["unload", "chat"]) == 0
    assert "chat" in capsys.readouterr().out

    class EmptyResponse(Response):
        def read(self):
            return b'{"unloaded": []}'

    monkeypatch.setattr(
        cli.urllib.request, "urlopen", lambda *_args, **_kwargs: EmptyResponse()
    )
    assert cli.main(["unload", "missing"]) == 1
    assert "not unloaded" in capsys.readouterr().err


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


def test_status_keeps_dead_gateway_view_after_pruning(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "owner_pid": os.getpid(),
            "gateway": {"pid": 999999, "port": 18000, "owner_pid": os.getpid()},
            "services": [],
        }),
        encoding="utf-8",
    )
    supervisor = Supervisor(state_path=state_path)

    status = supervisor.status()
    gateway = next(item for item in status.services if item["service"] == "gateway")

    assert gateway["running"] is False
    assert not state_path.exists()


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
    monkeypatch.setattr(cli, "is_windows", lambda: True)
    monkeypatch.setattr(cli, "record_gateway", lambda _pid, _port: None)
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)

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
    monkeypatch.setattr(cli, "is_windows", lambda: False)
    monkeypatch.setattr(cli, "record_gateway", lambda _pid, _port: None)
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)

    cli._launch_gateway(19000, detach=True)

    assert calls[0]["start_new_session"] is True
    assert calls[0]["close_fds"] is True


def test_detached_gateway_timeout_only_stops_owned_runtime(monkeypatch, tmp_path: Path) -> None:
    plan = build_plan(profile(32, (24,)), load_catalog(), Policy(roles=["chat"]))
    process = SimpleNamespace(pid=1234, terminate=lambda: None)
    down_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    monkeypatch.setattr(cli, "load_cache", dict)
    monkeypatch.setattr(cli, "bench_overlay", dict)
    monkeypatch.setattr(cli, "runtime_up", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "runtime_down", lambda *args, **kwargs: down_calls.append((args, kwargs)))
    monkeypatch.setattr(cli, "_launch_gateway", lambda _port, detach: (process, tmp_path / "gateway.log"))
    monkeypatch.setattr(cli, "_wait_gateway", lambda _port, _process: False)
    monkeypatch.setattr(cli, "clear_gateway", lambda _pid: None)

    result = cli._runtime(SimpleNamespace(
        command="up",
        port=18000,
        detach=True,
        dry_run=False,
        no_download=False,
        ignore_free_memory=False,
        json=True,
    ))

    assert result == 1
    assert down_calls == [((), {})]


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


def test_service_units_are_pure_and_platform_specific(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nmesh.runtime.service_unit.sys.executable", "/resolved/python")
    monkeypatch.setattr(service_unit_module, "nmesh_home", lambda: tmp_path)

    launcher_name, launcher = launcher_script(19000, "posix")
    assert launcher_name.endswith(".sh")
    assert launcher.startswith("#!/bin/sh\n")
    assert "\r" not in launcher
    assert "gateway.env" in launcher
    assert 'NMESH_HOME="$script_dir"' in launcher
    assert "exec /resolved/python -m nmesh.gateway.server --port 19000" in launcher

    filename, text, command = service_unit(19000, "posix")
    assert filename.endswith(".service")
    assert "ExecStart=" in text
    assert "nmesh-gateway-launcher.sh" in text
    assert command.startswith("systemctl --user")

    monkeypatch.setattr(service_unit_module.os, "name", "posix")
    monkeypatch.setattr(service_unit_module.sys, "platform", "darwin")
    filename, text, command = service_unit(19001)
    assert filename.endswith(".plist")
    assert "nmesh-gateway-launcher.sh" in text
    assert "<key>KeepAlive</key><true/>" in text
    assert command.startswith("launchctl")

    launcher_name, launcher = launcher_script(19002, "nt")
    assert launcher_name.endswith(".cmd")
    assert launcher.endswith("\r\n")
    assert "\n" not in launcher.replace("\r\n", "")
    assert "gateway.env" in launcher
    assert "set \"NMESH_HOME=%~dp0\"" in launcher
    assert "-m nmesh.gateway.server --port 19002" in launcher

    filename, text, command = service_unit(19002, "nt")
    assert filename.endswith(".cmd")
    assert "schtasks" in text
    assert "nmesh-gateway-launcher.cmd" in text


def test_autostart_install_writes_launcher_and_preserves_environment(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(service_unit_module, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "is_windows", lambda: True)
    monkeypatch.setenv("NMESH_API_KEY", "do-not-print")

    assert cli.main(["autostart", "--install", "--json"]) == 0
    output = capsys.readouterr().out
    data = json.loads(output)
    launcher_path = Path(data["launcher_path"])
    env_path = Path(data["env_path"])
    assert data["installed"] is True
    assert data["limitations"]
    assert "/sc onlogon" in data["limitations"][0]
    assert "self-crash" in data["limitations"][0]
    assert launcher_path.exists()
    assert env_path.read_text(encoding="utf-8") == (
        "NMESH_API_KEY=do-not-print\n"
        "# NMESH_HOME is set by the launcher to its own directory.\n"
    )
    assert "do-not-print" not in output

    env_path.write_text("NMESH_API_KEY=existing\n", encoding="utf-8")
    assert cli.main(["autostart", "--install", "--json"]) == 0
    capsys.readouterr()
    assert env_path.read_text(encoding="utf-8") == "NMESH_API_KEY=existing\n"

    assert cli.main(["autostart"]) == 0
    output = capsys.readouterr().out
    assert "/sc onlogon" in output
    assert "self-crash" in output
