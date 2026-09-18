from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import psutil

from nmesh import cli
from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan
from nmesh.runtime import service_unit as service_unit_module
from nmesh.runtime.logs import log_path, open_log, tail
from nmesh.runtime.service_unit import launcher_script, service_unit
from nmesh.runtime.supervisor import Supervisor

from .test_planner import profile


def test_gateway_state_is_terminated_and_removed(
    monkeypatch, tmp_path: Path,
) -> None:
    terminated: list[int] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: None,
    )
    supervisor = Supervisor(
        state_path=tmp_path / "state.json",
        terminator=terminated.append,
    )
    supervisor.record_gateway(os.getpid(), 18000)

    assert supervisor.down(foreign=True).running is False
    assert terminated == [os.getpid()]
    assert not (tmp_path / "state.json").exists()


def test_foreign_down_kills_orphaned_gateway_by_port(
    monkeypatch, tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "gateway": {"pid": 999999, "port": 18000,
                            "create_time": 1.0,
                            "owner_pid": os.getpid() + 1},
                "services": [],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: 4242,
    )
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    supervisor.down(foreign=True)

    assert terminated == [4242]
    assert not state_path.exists()


def test_foreign_down_sweeps_orphaned_gateway_without_state_entry(
    monkeypatch, tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"version": 2, "owner_pid": os.getpid() + 1, "services": []}),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: 4242,
    )
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    supervisor.down(foreign=True, gateway_port=18000)

    assert terminated == [4242]


def test_foreign_stop_gateway_kills_orphaned_gateway_by_port(
    monkeypatch, tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "gateway": {"pid": 999999, "port": 18000,
                            "create_time": 1.0,
                            "owner_pid": os.getpid() + 1},
                "services": [],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: 4242,
    )
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    assert supervisor.stop_gateway(foreign=True) is True
    assert terminated == [4242]


def test_unload_uses_persisted_plan_when_active_plan_lacks_service(
    monkeypatch, tmp_path: Path,
) -> None:
    """A gateway long outlives `nmesh up`, so its in-memory plan can miss a
    service that is recorded in state and still running."""
    launch = SimpleNamespace(
        health_url="http://127.0.0.1:1/health",
    )
    embed = SimpleNamespace(
        name="embed",
        port=18011,
        launch=launch,
        model_ref="model",
        quant="q8_0",
        backend="llamacpp",
        memory=SimpleNamespace(parallel_slots=1),
    )
    chat = SimpleNamespace(
        name="chat",
        port=18010,
        launch=launch,
        model_ref="model",
        quant="q4_k_m",
        backend="llamacpp",
        memory=SimpleNamespace(parallel_slots=1),
    )
    stale_plan = SimpleNamespace(services=[chat])
    persisted_plan = SimpleNamespace(services=[chat, embed])
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.load_plan", lambda: persisted_plan
    )
    pid = os.getpid()
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "services": [
                    {
                        "service": "embed",
                        "pid": pid,
                        "port": 18011,
                        "create_time": psutil.Process(pid).create_time(),
                        "external": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)
    supervisor.active_plan = stale_plan
    monkeypatch.setattr(Supervisor, "_healthy", lambda _self, _service: True)

    assert supervisor.unload("embed") is True
    assert terminated == [pid]
    assert supervisor.active_plan is persisted_plan


def test_status_merges_persisted_entries_beyond_in_memory(
    tmp_path: Path,
) -> None:
    """A supervisor tracking only some services in memory (e.g. the gateway,
    which never spawned the `up` services) must still report the rest of
    state.json instead of silently dropping live services."""
    pid = os.getpid()
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "services": [
                    {
                        "service": "embed",
                        "pid": pid,
                        "port": 18011,
                        "create_time": psutil.Process(pid).create_time(),
                        "external": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    supervisor = Supervisor(state_path=state_path)
    supervisor.adopted["chat"] = {"pid": pid, "port": 18010}

    names = {str(item["service"]) for item in supervisor.status().services}

    assert names == {"chat", "embed"}


def _up_service(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        port=18010,
        backend="llamacpp",
        model_ref="model",
        quant="q4_k_m",
        launch=SimpleNamespace(health_url=None, shared_daemon=False, argv=[]),
        memory=SimpleNamespace(parallel_slots=1),
    )


def test_up_does_not_persist_admission_dropped_plan(
    monkeypatch, tmp_path: Path,
) -> None:
    """Admission may run a subset under memory pressure, but the saved plan
    must keep the user's full service set — the drop is transient."""
    chat = _up_service("chat")
    embed = _up_service("embed")
    plan = SimpleNamespace(
        services=[chat, embed], warnings=[], swap_group=set(), policy=None,
    )
    reduced = SimpleNamespace(
        services=[chat], warnings=[], swap_group=set(), policy=None,
    )
    saved: list[SimpleNamespace] = []
    monkeypatch.setattr("nmesh.runtime.supervisor.save_plan", saved.append)
    supervisor = Supervisor(state_path=tmp_path / "state.json")
    monkeypatch.setattr(supervisor, "_admit", lambda _plan, _cache: reduced)
    monkeypatch.setattr(supervisor, "_adopt", lambda _service: False)
    monkeypatch.setattr(supervisor, "_already_up", lambda _service: False)
    monkeypatch.setattr(supervisor, "_wait_health", lambda _service: True)
    supervisor.launcher = lambda _service: SimpleNamespace(
        pid=999999,
        poll=lambda: 0,
        wait=lambda *a, **k: None,
        terminate=lambda: None,
        kill=lambda: None,
    )

    supervisor.up(plan, no_download=True)
    supervisor.disarm_atexit()

    assert set(supervisor.processes) == {"chat"}
    assert saved == []


def test_up_surfaces_admission_warnings(
    monkeypatch, tmp_path: Path,
) -> None:
    """Admission may drop services under memory pressure — the warnings it
    produced must reach the caller, not stay hidden inside the runtime."""
    chat = _up_service("chat")
    embed = _up_service("embed")
    plan = SimpleNamespace(
        services=[chat, embed], warnings=[], swap_group=set(), policy=None,
    )
    reduced = SimpleNamespace(
        services=[chat],
        warnings=["admission dropped embed: not enough free memory"],
        swap_group=set(),
        policy=None,
    )
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.save_plan", lambda _plan: None
    )
    supervisor = Supervisor(state_path=tmp_path / "state.json")
    monkeypatch.setattr(supervisor, "_admit", lambda _plan, _cache: reduced)
    monkeypatch.setattr(supervisor, "_adopt", lambda _service: False)
    monkeypatch.setattr(supervisor, "_already_up", lambda _service: False)
    monkeypatch.setattr(supervisor, "_wait_health", lambda _service: True)
    supervisor.launcher = lambda _service: SimpleNamespace(
        pid=999999,
        poll=lambda: 0,
        wait=lambda *a, **k: None,
        terminate=lambda: None,
        kill=lambda: None,
    )

    result = supervisor.up(plan, no_download=True)
    supervisor.disarm_atexit()

    assert set(supervisor.processes) == {"chat"}
    assert result.warnings == [
        "admission dropped embed: not enough free memory"
    ]


def test_up_writes_plan_to_configured_plan_path(
    monkeypatch, tmp_path: Path,
) -> None:
    """A supervisor with an alternate plan path (e.g. `spec measure`'s
    temporary one) must not clobber the user's real plan.json."""
    chat = _up_service("chat")
    plan = SimpleNamespace(
        services=[chat], warnings=[], swap_group=set(), policy=None,
    )
    saved: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.save_plan",
        lambda _plan, path=None: saved.append((_plan, path)),
    )
    plan_path = tmp_path / "plan.json"
    supervisor = Supervisor(
        state_path=tmp_path / "state.json", plan_path=plan_path
    )
    monkeypatch.setattr(supervisor, "_adopt", lambda _service: False)
    monkeypatch.setattr(supervisor, "_already_up", lambda _service: False)
    monkeypatch.setattr(supervisor, "_wait_health", lambda _service: True)
    supervisor.launcher = lambda _service: SimpleNamespace(
        pid=999999,
        poll=lambda: 0,
        wait=lambda *a, **k: None,
        terminate=lambda: None,
        kill=lambda: None,
    )

    supervisor.up(plan, no_download=True, admit=False)
    supervisor.disarm_atexit()

    assert saved == [(plan, plan_path)]


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


def test_unhealthy_message_names_occupied_port(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        message = Supervisor()._unhealthy_message("chat", port)
    finally:
        listener.close()

    assert "Service did not become healthy: chat" in message
    assert f"port {port}" in message


def test_unhealthy_message_omits_port_hint_when_free(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()

    message = Supervisor()._unhealthy_message("chat", free_port)

    assert "port" not in message.lower()


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


def test_gateway_creation_mismatch_is_pruned_without_termination(
    monkeypatch, tmp_path: Path,
) -> None:
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
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: None,
    )
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


def test_launch_gateway_adopts_healthy_existing_gateway(
    monkeypatch, tmp_path: Path,
) -> None:
    recorded: list[tuple[int, int]] = []

    def popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Popen must not run when a gateway is already healthy")

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "gateway_health", lambda _port: True)
    monkeypatch.setattr(cli, "gateway_listener_pid", lambda _port: 4242)
    monkeypatch.setattr(
        cli, "record_gateway", lambda pid, port: recorded.append((pid, port)),
    )
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)

    process, log_path = cli._launch_gateway(19000, detach=True)

    assert isinstance(process, cli._AdoptedGateway)
    assert process.pid == 4242
    assert log_path is None
    assert recorded == [(4242, 19000)]


def test_launch_gateway_refuses_unknown_port_owner(
    monkeypatch, tmp_path: Path,
) -> None:
    def popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Popen must not run when the port is already served")

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "gateway_health", lambda _port: True)
    monkeypatch.setattr(cli, "gateway_listener_pid", lambda _port: None)
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)

    try:
        cli._launch_gateway(19000, detach=True)
    except OSError as error:
        assert "19000" in str(error)
    else:
        raise AssertionError("expected OSError for an unclaimed healthy port")


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


def test_autostart_install_writes_unit_file(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(service_unit_module, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "is_windows", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")
    config_home = tmp_path / "xdg-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    assert cli.main(["autostart", "--install", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    unit_path = Path(data["unit_path"])
    assert unit_path == config_home / "systemd" / "user" / "nmesh-gateway.service"
    unit = unit_path.read_text(encoding="utf-8")
    assert "[Service]" in unit
    assert "ExecStart=" in unit
    assert data["install_command"].endswith(str(unit_path))


def test_status_cli_renders_table_not_repr(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    assert cli.main(["status", "--port", "19999"]) == 0
    out = capsys.readouterr().out
    assert "RuntimeStatus(" not in out
    assert "gateway" in out and "stopped" in out


def test_down_cli_no_services_message(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    # down sweeps the gateway port even for foreign homes — point it at an
    # unused port so a developer's live stack doesn't leak into the test.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    free_port = sock.getsockname()[1]
    sock.close()

    assert cli.main(["down", "--port", str(free_port)]) == 0
    out = capsys.readouterr().out
    assert "RuntimeStatus(" not in out
    assert "no services running" in out


def test_up_status_entries_include_port(
    monkeypatch, tmp_path: Path,
) -> None:
    """The `up` result feeds the Services table — entries must carry the
    service port or the row renders a blank Port cell until `status`."""
    chat = _up_service("chat")
    plan = SimpleNamespace(
        services=[chat], warnings=[], swap_group=set(), policy=None,
    )
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.save_plan", lambda *_a: None
    )
    supervisor = Supervisor(state_path=tmp_path / "state.json")
    monkeypatch.setattr(supervisor, "_admit", lambda _plan, _cache: _plan)
    monkeypatch.setattr(supervisor, "_adopt", lambda _service: False)
    monkeypatch.setattr(supervisor, "_already_up", lambda _service: False)
    monkeypatch.setattr(supervisor, "_wait_health", lambda _service: True)
    supervisor.launcher = lambda _service: SimpleNamespace(
        pid=999999,
        poll=lambda: 0,
        wait=lambda *a, **k: None,
        terminate=lambda: None,
        kill=lambda: None,
    )

    result = supervisor.up(plan, no_download=True)
    supervisor.disarm_atexit()

    entry = next(s for s in result.services if s.get("service") == "chat")
    assert entry.get("port") == 18010


def test_down_reports_stopped_services(
    monkeypatch, tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "owner_pid": os.getpid() + 1,
                "services": [
                    {
                        "service": "chat",
                        "pid": 123,
                        "port": 18010,
                        "model_ref": "model.gguf",
                        "backend": "llamacpp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        "nmesh.runtime.supervisor._pid_alive", lambda *_a: True,
    )
    monkeypatch.setattr(
        "nmesh.runtime.supervisor.gateway_listener_pid", lambda _port: None,
    )
    supervisor = Supervisor(state_path=state_path, terminator=terminated.append)

    result = supervisor.down(foreign=True)

    assert terminated == [123]
    services = {item["service"]: item for item in result.services}
    assert services["chat"]["running"] is False
    assert services["chat"]["port"] == 18010


def test_jobs_cli_lists_gateway_jobs(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    payload = {
        "jobs": [
            {
                "id": "job-1",
                "service": "chat",
                "endpoint": "/v1/chat/completions",
                "state": "running",
                "queued_at": 1.0,
                "started_at": 2.0,
                "finished_at": None,
                "detail": None,
            }
        ],
        "counts": {"chat": {"queued": 0, "running": 1}},
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        "nmesh.cli.urllib.request.urlopen", lambda *_a, **_k: _Response()
    )
    assert cli.main(["jobs", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"job-1"' in out
    assert cli.main(["jobs"]) == 0
    assert "job-1" in capsys.readouterr().out


def test_jobs_cli_gateway_unreachable(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr("nmesh.cli.urllib.request.urlopen", _raise)
    assert cli.main(["jobs"]) == 1
    captured = capsys.readouterr()
    assert "18000" in captured.err
    assert "nmesh up" in captured.err
def test_jobs_cli_old_gateway_404(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    def _raise(*_a, **_k):
        raise HTTPError("http://x", 404, "not found", {}, None)

    monkeypatch.setattr("nmesh.cli.urllib.request.urlopen", _raise)
    assert cli.main(["jobs"]) == 1
    captured = capsys.readouterr()
    assert "older build" in captured.err or "古いビルド" in captured.err
    assert "nmesh down" in captured.err


def test_jobs_cli_cancel(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    seen: list[object] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {"id": "job-2", "state": "cancelled"}
            ).encode()

    def _open(request, *_a, **_k):
        seen.append(getattr(request, "method", "GET"))
        return _Response()

    monkeypatch.setattr("nmesh.cli.urllib.request.urlopen", _open)
    assert cli.main(["jobs", "--cancel", "job-2"]) == 0
    assert seen == ["DELETE"]
    assert "job-2" in capsys.readouterr().out


def test_jobs_cli_cancel_conflict(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    def _raise(*_a, **_k):
        raise HTTPError("http://x", 409, "conflict", {}, None)

    monkeypatch.setattr("nmesh.cli.urllib.request.urlopen", _raise)
    assert cli.main(["jobs", "--cancel", "job-1"]) == 1
    assert "queued" in capsys.readouterr().err

def test_run_cli_unreachable_gateway_hint(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    assert cli.main(["run", "hi", "--port", "19999"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "gateway unavailable" in captured.err
    assert "nmesh up" in captured.err
