from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import psutil
import pytest

from nmesh import i18n
from nmesh.bench import benchmark, benchmark_key, load_cache, save_cache
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.gateway import estimate_tokens, route
from nmesh.planner import Policy, build_plan, free_budgets
from nmesh.planner import save_plan as planner_save_plan
from nmesh.probe import HardwareProfile
from nmesh.runtime import Supervisor
from nmesh.runtime import supervisor as supervisor_module
from nmesh.runtime.acquisition import Acquired

from .test_planner import profile


@pytest.fixture
def catalog() -> list[ModelSpec]:
    return load_catalog()


@pytest.fixture(autouse=True)
def isolate_saved_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "save_plan",
        lambda plan: planner_save_plan(plan, tmp_path / "plan.json"),
    )


def test_route_rules(catalog: list[object]) -> None:
    plan = build_plan(profile(64, (24,)), catalog, Policy(roles=["chat", "code", "embed"]))
    assert route({"tools": [{"type": "function"}]}, plan) == plan.routing.role_to_service["chat"]
    assert route({"messages": [{"content": "```python\nprint(1)\n```"}]}, plan) == (
        plan.routing.role_to_service["code"]
    )
    assert route({"messages": [{"content": "hello"}]}, plan) == plan.routing.role_to_service["chat"]


def test_estimate_tokens_is_script_aware() -> None:
    japanese = "日本語" * 100
    ascii_text = "abcd" * 100
    assert estimate_tokens(japanese) >= len(japanese) * 0.9
    assert estimate_tokens(ascii_text) == len(ascii_text) // 4


def test_route_uses_script_aware_context_and_reserved_output(catalog) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    base = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    small = replace(base.services[0], name="small", context=1024)
    large = replace(base.services[0], name="large", context=4096)
    plan = replace(
        base,
        services=[small, large],
        routing=replace(
            base.routing,
            role_to_service={"chat": "small"},
            aliases={"nmesh-auto": "small"},
        ),
    )
    japanese = "日本語" * 300
    assert len(japanese) // 4 < small.context * 0.8
    assert route({"messages": [{"content": japanese}]}, plan) == "large"
    borderline = "abcd" * 650
    assert len(borderline) // 4 < small.context * 0.8
    assert route({
        "messages": [{"content": borderline}],
        "max_tokens": 300,
    }, plan) == "large"


def test_bench_cache_round_trip(tmp_path) -> None:
    key = benchmark_key("model", "q4_k_m", "llamacpp", "cpu", 0)
    path = tmp_path / "bench.json"
    save_cache({key: benchmark(lambda _prefill, _decode: 3.0)}, path)
    assert load_cache(path)[key] == 3.0


def test_benchmark_key_preserves_f16_and_separates_q8() -> None:
    old_key = "model|q4_k_m|llamacpp|cpu|0"
    f16_key = benchmark_key("model", "q4_k_m", "llamacpp", "cpu", 0, "f16")
    q8_key = benchmark_key("model", "q4_k_m", "llamacpp", "cpu", 0, "q8_0")
    assert f16_key == old_key
    assert q8_key == f"{old_key}|kvq8_0"
    assert q8_key != f16_key


def test_supervisor_fallback_with_fake_launcher(tmp_path, catalog: list[object]) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(plan.services[0], launch=replace(plan.services[0].launch, health_url=None))
    plan = replace(plan, services=[service])
    calls: list[str] = []

    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def launcher(item) -> FakeProcess:
        calls.append(item.quant)
        if len(calls) == 1:
            raise RuntimeError("synthetic startup failure")
        return FakeProcess()

    supervisor = Supervisor(launcher, tmp_path / "state.json", health_timeout=0.01)
    result = supervisor.up(plan, no_download=True, admit=False)
    assert result.running
    assert len(calls) == 2
    supervisor.down()


def test_supervisor_fallback_skips_unknown_artifact_quant(tmp_path, catalog) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        quant="iq3_m",
        launch=replace(plan.services[0].launch, health_url=None),
    )
    supervisor = Supervisor(state_path=tmp_path / "state.json")

    skipped = supervisor._fallback(replace(plan, services=[service]), 1)

    assert skipped.services[0].quant == "iq3_m"
    warning = i18n.t(
        "warn.quant_fallback_skipped",
        "en",
        service=service.name,
        quant="iq3_m",
    )
    assert skipped.warnings.count(warning) == 1

    ordinary = supervisor._fallback(
        replace(plan, services=[replace(service, quant="q4_k_m")]), 1
    )
    assert ordinary.services[0].quant == "q4_0"


def test_supervisor_unload_adopts_live_state_process(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    planner_save_plan(plan)
    supervisor = Supervisor(state_path=tmp_path / "state.json")
    (tmp_path / "state.json").write_text(
        json.dumps({"services": [{"service": "chat", "pid": 1234}]}),
        encoding="utf-8",
    )
    terminated: list[int] = []
    monkeypatch.setattr(supervisor, "_entry_alive", lambda _entry: True)
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)
    monkeypatch.setattr(supervisor, "_terminator", terminated.append)

    assert supervisor.unload("chat")
    assert terminated == [1234]


def test_supervisor_dry_run_contains_argv(tmp_path, catalog: list[object]) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    result = Supervisor(state_path=tmp_path / "state.json").up(plan, dry_run=True)
    assert result.services
    assert result.services[0]["argv"] == plan.services[0].launch.argv


def test_supervisor_rewrites_acquired_model_and_records_note(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url=None),
    )
    plan = replace(plan, services=[service])
    acquired_path = tmp_path / "actual-q2_k.gguf"
    monkeypatch.setattr(
        supervisor_module,
        "acquire",
        lambda _service: Acquired(acquired_path, "q2_k", True),
    )
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "acquired-state.json",
        health_timeout=0.01,
    )
    result = supervisor.up(plan, admit=False)
    launched = supervisor.active_plan.services[0]
    assert result.running
    assert launched.model_ref == str(acquired_path)
    assert launched.quant == "q2_k"
    assert launched.launch.argv[launched.launch.argv.index("-m") + 1] == str(acquired_path)
    assert result.services[0]["note"] == f"{service.quant} -> q2_k"
    payload = json.loads((tmp_path / "acquired-state.json").read_text(encoding="utf-8"))
    assert payload["services"][0]["model_ref"] == str(acquired_path)
    assert payload["services"][0]["quant"] == "q2_k"
    assert payload["services"][0]["note"] == f"{service.quant} -> q2_k"
    supervisor.down()


def test_supervisor_rechecks_acquired_artifact_bytes(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    hardware = profile(32, (24,))
    plan = build_plan(hardware, catalog, Policy(roles=["chat"]))
    service = plan.services[0]
    supervisor = Supervisor(
        state_path=tmp_path / "artifact-state.json",
        probe=lambda: hardware,
        catalog=lambda: catalog,
    )
    updated, actualized, changed = supervisor._apply_acquired(
        plan,
        service,
        Acquired(None, None, False, artifact_bytes=int(service.memory.weight_bytes * 2)),
    )
    assert changed
    assert actualized.memory.weight_bytes == pytest.approx(
        service.memory.weight_bytes * 2
    )
    assert any("real artifact bytes exceeded" in warning for warning in updated.warnings)

    unchanged, _, changed = supervisor._apply_acquired(
        plan,
        service,
        Acquired(None, None, False, artifact_bytes=int(service.memory.weight_bytes)),
    )
    assert not changed
    assert not any("real artifact bytes exceeded" in warning
                   for warning in unchanged.warnings)

    near, _, changed = supervisor._apply_acquired(
        plan,
        service,
        Acquired(None, None, False, artifact_bytes=int(service.memory.weight_bytes * 1.05)),
    )
    assert changed
    assert not any("real artifact bytes exceeded" in warning
                   for warning in near.warnings)


def test_supervisor_applies_ollama_derived_model_ref(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        backend="ollama",
        model_ref="qwen2.5:0.5b-instruct",
        launch=replace(plan.services[0].launch, argv=["ollama", "serve"], health_url=None),
    )
    plan = replace(plan, services=[service])
    monkeypatch.setattr(
        supervisor_module,
        "acquire",
        lambda _service: Acquired(
            None, None, False, model_ref="nmesh-chat-c8192"
        ),
    )
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "ollama-state.json",
        health_timeout=0.01,
    )

    result = supervisor.up(plan, admit=False)

    assert result.running
    launched = supervisor.active_plan.services[0]
    assert launched.model_ref == "nmesh-chat-c8192"
    assert launched.launch.argv == ["ollama", "serve"]
    supervisor.down()


def test_supervisor_ollama_create_failure_keeps_base_and_warns(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        backend="ollama",
        model_ref="qwen2.5:0.5b-instruct",
        launch=replace(plan.services[0].launch, argv=["ollama", "serve"], health_url=None),
    )
    plan = replace(plan, services=[service])
    warning = "Ollama context could not be set for chat; it will run at Ollama's default context, not the planned 8192."
    monkeypatch.setattr(
        supervisor_module,
        "acquire",
        lambda _service: Acquired(
            None,
            None,
            False,
            model_ref=service.model_ref,
            warning=warning,
        ),
    )
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "ollama-failure-state.json",
        health_timeout=0.01,
    )

    result = supervisor.up(plan, admit=False)

    assert result.running
    assert supervisor.active_plan.services[0].model_ref == service.model_ref
    assert warning in supervisor.active_plan.warnings
    assert result.services[0]["note"] == warning
    supervisor.down()


def test_supervisor_no_download_ollama_warns_and_keeps_base(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(
        plan.services[0],
        backend="ollama",
        model_ref="qwen2.5:0.5b-instruct",
        launch=replace(plan.services[0].launch, argv=["ollama", "serve"], health_url=None),
    )
    plan = replace(plan, services=[service])
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "ollama-no-download-state.json",
        health_timeout=0.01,
    )

    result = supervisor.up(plan, no_download=True, admit=False)

    assert result.running
    assert supervisor.active_plan.services[0].model_ref == service.model_ref
    assert any("default context" in warning for warning in supervisor.active_plan.warnings)
    supervisor.down()


class _AdmissionProcess:
    pid = 456

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _RecoverProcess:
    _next_pid = 1000

    def __init__(self) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.exit_code or 0


def _recovery_plan(catalog: list[ModelSpec]):
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    service = replace(plan.services[0], launch=replace(plan.services[0].launch, health_url=None))
    return replace(plan, services=[service])


def test_supervisor_heartbeat_revives_dead_process(tmp_path, catalog: list[ModelSpec]) -> None:
    plan = _recovery_plan(catalog)
    processes: list[_RecoverProcess] = []
    supervisor = Supervisor(
        lambda _service: processes.append(_RecoverProcess()) or processes[-1],
        tmp_path / "heartbeat.json",
        health_timeout=0.01,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    supervisor.up(plan, no_download=True, admit=False)
    processes[0].exit_code = 137
    result = supervisor.heartbeat()
    assert len(processes) == 2
    assert result.services[0]["restarts"] == 1
    supervisor.down()


def test_supervisor_heartbeat_restart_budget_marks_failure(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = _recovery_plan(catalog)
    processes: list[_RecoverProcess] = []
    supervisor = Supervisor(
        lambda _service: processes.append(_RecoverProcess()) or processes[-1],
        tmp_path / "budget.json",
        health_timeout=0.01,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    supervisor.up(plan, no_download=True, admit=False)
    for _ in range(4):
        processes[-1].exit_code = 137
        supervisor.heartbeat()
    assert len(processes) == 4
    failed = next(item for item in supervisor.status().services if item["service"] == "chat")
    assert failed["running"] is False
    assert "failed" in failed
    supervisor.down()


def test_supervisor_heartbeat_skips_unloaded_swap_member(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = _recovery_plan(catalog)
    plan = replace(plan, swap_group=["chat"])
    calls: list[str] = []
    supervisor = Supervisor(
        lambda service: calls.append(service.name) or _AdmissionProcess(),
        tmp_path / "swap-heartbeat.json",
        health_timeout=0.01,
    )
    supervisor.active_plan = plan
    supervisor.heartbeat()
    assert calls == []
    supervisor.down()


def test_supervisor_idle_unload_does_not_restart_and_revives(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = _recovery_plan(catalog)
    processes: list[_RecoverProcess] = []
    supervisor = Supervisor(
        lambda _service: processes.append(_RecoverProcess()) or processes[-1],
        tmp_path / "idle.json",
        health_timeout=0.01,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    supervisor.up(plan, no_download=True, admit=False)
    supervisor.restarts["chat"] = [1.0]

    assert supervisor.unload("missing") is False
    supervisor.shared_services.add("chat")
    assert supervisor.unload("chat") is False
    supervisor.shared_services.clear()
    supervisor.external_shared.add("chat")
    assert supervisor.unload("chat") is False
    supervisor.external_shared.clear()

    assert supervisor.unload("chat") is True
    assert processes[0].poll() == 0
    assert supervisor.idle_services() == {"chat"}
    assert supervisor.restarts == {}
    idle = next(item for item in supervisor.status().services if item["service"] == "chat")
    assert idle["idle"] is True
    assert idle["running"] is False
    assert "failed" not in idle
    assert supervisor.status().running is False
    supervisor.heartbeat()
    assert len(processes) == 1

    supervisor.ensure_running("chat", plan)
    assert len(processes) == 2
    assert supervisor.idle_services() == set()
    supervisor.down()


def test_supervisor_ensure_running_revives_dead_process(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = _recovery_plan(catalog)
    processes: list[_RecoverProcess] = []
    supervisor = Supervisor(
        lambda _service: processes.append(_RecoverProcess()) or processes[-1],
        tmp_path / "ensure.json",
        health_timeout=0.01,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    supervisor.up(plan, no_download=True, admit=False)
    processes[0].exit_code = 1
    supervisor.ensure_running("chat", plan)
    assert len(processes) == 2
    assert supervisor.restarts["chat"]
    supervisor.down()


def test_supervisor_adopts_healthy_external_service(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    plan = replace(plan, services=[service])
    calls: list[str] = []
    process = _AdmissionProcess()
    supervisor = Supervisor(
        lambda item: calls.append(item.name) or process,
        tmp_path / "adopt.json",
        health_timeout=0.01,
    )
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)
    supervisor.ensure_running("chat", plan)
    assert calls == []
    assert supervisor.external_shared == {"chat"}
    result = supervisor.status()
    assert result.services[0]["external"] is True
    assert supervisor.unload("chat") is False
    supervisor.down()
    assert process.poll() is None


def test_supervisor_adopts_recorded_pid_and_unloads_it(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    plan = replace(plan, services=[service])
    state_path = tmp_path / "adopted.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "services": [{
                "service": "chat",
                "pid": os.getpid(),
                "create_time": psutil.Process(os.getpid()).create_time(),
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(
        lambda _service: pytest.fail("recorded process should be adopted"),
        state_path,
        terminator=terminated.append,
    )
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)

    supervisor.ensure_running("chat", plan)

    assert supervisor.adopted["chat"]["pid"] == os.getpid()
    adopted_status = next(
        item for item in supervisor.status().services if item["service"] == "chat"
    )
    assert adopted_status["pid"] == os.getpid()
    assert adopted_status["running"] is True
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["services"][0]["pid"] == os.getpid()
    assert payload["services"][0]["adopted"] is True
    assert supervisor.unload("chat") is True
    assert terminated == [os.getpid()]
    assert supervisor.idle_services() == {"chat"}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["services"] == []


def test_supervisor_heartbeat_relaunches_dead_adopted_service(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    plan = replace(plan, services=[service])
    state_path = tmp_path / "dead-adopted.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "services": [{
                "service": "chat",
                "pid": 1234,
                "create_time": 1.0,
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    alive = True
    launched: list[str] = []

    def pid_alive(pid: int, create_time: float | None = None) -> bool:
        return pid == 1234 and alive

    supervisor = Supervisor(
        lambda item: launched.append(item.name) or _RecoverProcess(),
        state_path,
        health_timeout=0.01,
    )
    monkeypatch.setattr(supervisor_module, "_pid_alive", pid_alive)
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)
    supervisor.ensure_running("chat", plan)
    alive = False

    supervisor.heartbeat()

    assert launched == ["chat"]
    assert supervisor.adopted == {}
    assert supervisor.processes["chat"].poll() is None
    supervisor.down()


def test_supervisor_down_stops_adopted_service(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    plan = replace(plan, services=[service])
    state_path = tmp_path / "down-adopted.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "services": [{
                "service": "chat",
                "pid": os.getpid(),
                "create_time": psutil.Process(os.getpid()).create_time(),
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(
        lambda _service: pytest.fail("recorded process should be adopted"),
        state_path,
        terminator=terminated.append,
    )
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)

    supervisor.ensure_running("chat", plan)
    supervisor.down()

    assert terminated == [os.getpid()]
    assert not state_path.exists()


def test_supervisor_status_state_fallback_is_running(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"services": [{
            "service": "chat",
            "pid": os.getpid(),
            "port": 18010,
        }]}),
        encoding="utf-8",
    )
    result = Supervisor(state_path=state_path).status()
    assert result.running is True
    assert result.services[0]["service"] == "chat"


class _StateProcess:
    pid = os.getpid()

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_supervisor_state_is_shared_between_processes(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    plan = _recovery_plan(catalog)
    state_path = tmp_path / "shared-state.json"
    first = Supervisor(lambda _service: _StateProcess(), state_path, health_timeout=0.01)
    first.up(plan, no_download=True, admit=False)
    second = Supervisor(state_path=state_path)
    result = second.status()
    assert result.running
    assert result.services[0]["running"] is True
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["owner_pid"] == os.getpid()
    assert payload["services"][0]["health_url"] is None
    first.down()


def test_supervisor_state_prunes_dead_entries(tmp_path) -> None:
    state_path = tmp_path / "dead-state.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "owner_pid": 123,
            "services": [{
                "service": "chat",
                "pid": os.getpid(),
                "create_time": time.time() + 100,
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    supervisor = Supervisor(state_path=state_path)
    result = supervisor.status()
    assert result.running is False
    assert result.services[0]["running"] is False
    assert not state_path.exists()


def test_supervisor_status_does_not_arm_atexit(tmp_path) -> None:
    state_path = tmp_path / "status-only.json"
    state_path.write_text(
        '{"version": 2, "owner_pid": 1, "services": []}',
        encoding="utf-8",
    )
    supervisor = Supervisor(state_path=state_path)
    supervisor.status()
    assert supervisor._atexit_armed is False
    assert state_path.exists()


def test_supervisor_foreign_down_terminates_recorded_pid(tmp_path) -> None:
    state_path = tmp_path / "foreign-state.json"
    state_path.write_text(
        json.dumps({
            "version": 2,
            "owner_pid": os.getpid() + 1,
            "services": [{
                "service": "chat",
                "pid": os.getpid(),
                "create_time": psutil.Process(os.getpid()).create_time(),
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(
        state_path=state_path,
        terminator=terminated.append,
    )
    supervisor.down()
    assert terminated == []
    assert state_path.exists()
    supervisor.down(foreign=True)
    assert terminated == [os.getpid()]
    assert not state_path.exists()


def test_supervisor_loads_v1_state(tmp_path) -> None:
    state_path = tmp_path / "v1-state.json"
    state_path.write_text(
        json.dumps({
            "services": [{
                "service": "chat",
                "pid": os.getpid(),
                "port": 18010,
            }],
        }),
        encoding="utf-8",
    )
    terminated: list[int] = []
    supervisor = Supervisor(
        state_path=state_path,
        terminator=terminated.append,
    )
    result = supervisor.status()
    assert result.running
    assert result.services[0]["running"] is True
    supervisor.down(foreign=True)
    assert terminated == [os.getpid()]


def test_supervisor_heartbeat_adopts_healthy_service_past_restart_budget(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    supervisor = Supervisor(
        lambda _service: pytest.fail("healthy external service should not launch"),
        tmp_path / "adopt-budget.json",
        health_timeout=0.01,
    )
    supervisor.active_plan = replace(plan, services=[service])
    supervisor.restarts[service.name] = [time.monotonic()] * supervisor_module.MAX_RESTARTS
    supervisor.failed[service.name] = "stale failure"
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)
    supervisor.heartbeat()
    assert service.name in supervisor.external_shared
    assert supervisor.failed == {}


def test_supervisor_boot_heartbeat_recovers_resident_and_persisted_services(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    base = _recovery_plan(catalog).services[0]
    resident = replace(base, name="resident", resident=True)
    persisted = replace(base, name="persisted", resident=False)
    lazy = replace(base, name="lazy", resident=False)
    plan = replace(_recovery_plan(catalog), services=[resident, persisted, lazy])
    planner_save_plan(plan, tmp_path / "plan.json")
    (tmp_path / "state.json").write_text(
        json.dumps({"version": 2, "services": [{"service": "persisted"}]}),
        encoding="utf-8",
    )
    launched: list[str] = []

    class Process:
        pid = 1234

        def poll(self):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout=None):
            return None

    supervisor = Supervisor(
        lambda service: launched.append(service.name) or Process(),
        tmp_path / "state.json",
        health_timeout=0.01,
    )
    supervisor._wait_health = lambda _service, timeout=None: True

    supervisor.heartbeat()

    assert launched == ["resident", "persisted"]
    assert supervisor.active_plan is not None
    assert "lazy" not in supervisor.processes
    supervisor.ensure_running("resident")
    supervisor.heartbeat()
    assert launched == ["resident", "persisted"]
    supervisor.down()


def test_supervisor_boot_heartbeat_adopts_listening_service(
    tmp_path, catalog: list[ModelSpec], monkeypatch
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    plan = _recovery_plan(catalog)
    service = replace(
        plan.services[0],
        resident=False,
        launch=replace(plan.services[0].launch, health_url="http://127.0.0.1:1/health"),
    )
    planner_save_plan(replace(plan, services=[service]), tmp_path / "plan.json")
    supervisor = Supervisor(
        lambda _service: pytest.fail("listening service should be adopted"),
        tmp_path / "state.json",
        health_timeout=0.01,
    )
    monkeypatch.setattr(supervisor, "_healthy", lambda _service: True)

    supervisor.heartbeat()

    assert supervisor.external_shared == {service.name}
    supervisor.down()


def test_supervisor_boot_heartbeat_without_plan_returns_status(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    supervisor = Supervisor(state_path=tmp_path / "state.json")

    result = supervisor.heartbeat()

    assert supervisor.active_plan is None
    assert result.running is False


def test_supervisor_admission_replans_against_free_memory(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    base = profile(32, (24,))
    plan = build_plan(base, catalog, Policy(roles=["chat"]))
    starved = replace(
        base,
        available_ram_bytes=16 * 1024**3,
        gpus=[replace(base.gpus[0], free_vram_bytes=8 * 1024**3)],
    )
    plan_path = tmp_path / "plan.json"
    planner_save_plan(plan, plan_path)
    original_plan = plan_path.read_bytes()
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "admit.json",
        health_timeout=0.01,
        probe=lambda: starved,
        catalog=lambda: catalog,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    result = supervisor.up(plan, no_download=True)
    assert result.running
    assert supervisor.active_plan is not None
    assert any("Free-memory admission" in warning for warning in supervisor.active_plan.warnings)
    assert supervisor.active_plan.services != plan.services
    assert supervisor.active_plan.services[0].memory.gpu_bytes <= free_budgets(starved)[0] + 1
    assert plan_path.read_bytes() == original_plan
    supervisor.down()


def test_supervisor_admission_can_be_skipped_or_fail_open(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    base = profile(32, (24,))
    plan = build_plan(base, catalog, Policy(roles=["chat"]))
    calls = 0

    def probe() -> HardwareProfile:
        nonlocal calls
        calls += 1
        raise RuntimeError("probe unavailable")

    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "skip.json",
        health_timeout=0.01,
        probe=probe,
        catalog=lambda: catalog,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    result = supervisor.up(plan, no_download=True, admit=False)
    assert result.running
    assert calls == 0
    supervisor.down()

    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "fail-open.json",
        health_timeout=0.01,
        probe=probe,
        catalog=lambda: catalog,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    result = supervisor.up(plan, no_download=True)
    assert result.running
    assert supervisor.active_plan is not None
    assert any("admission skipped" in warning for warning in supervisor.active_plan.warnings)
    supervisor.down()


def test_supervisor_admission_leaves_roomy_plan_unchanged(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    roomy = profile(32, (24,))
    plan = build_plan(roomy, catalog, Policy(roles=["chat"]))
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "roomy.json",
        health_timeout=0.01,
        probe=lambda: roomy,
        catalog=lambda: catalog,
    )
    supervisor._wait_health = lambda _service, timeout=None: True
    result = supervisor.up(plan, no_download=True)
    assert result.running
    assert supervisor.active_plan is not None
    assert supervisor.active_plan.services == plan.services
    supervisor.down()


def test_supervisor_admission_excludes_already_up_services(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    profile_now = profile(32, (24,))
    plan = build_plan(profile_now, catalog, Policy(roles=["chat"]))
    supervisor = Supervisor(
        lambda _service: _AdmissionProcess(),
        tmp_path / "already-up.json",
        health_timeout=0.01,
        probe=lambda: replace(
            profile_now,
            available_ram_bytes=1 * 1024**3,
            gpus=[replace(profile_now.gpus[0], free_vram_bytes=1 * 1024**3)],
        ),
        catalog=lambda: catalog,
    )
    supervisor.processes[plan.services[0].name] = _AdmissionProcess()
    assert supervisor._admit(plan) is plan
    supervisor.down()
