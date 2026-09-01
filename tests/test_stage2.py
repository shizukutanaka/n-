from __future__ import annotations

from dataclasses import replace

import pytest

from nmesh.bench import benchmark, benchmark_key, load_cache, save_cache
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.gateway import route
from nmesh.planner import Policy, build_plan, free_budgets
from nmesh.planner import save_plan as planner_save_plan
from nmesh.probe import HardwareProfile
from nmesh.runtime import Supervisor
from nmesh.runtime import supervisor as supervisor_module

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


def test_bench_cache_round_trip(tmp_path) -> None:
    key = benchmark_key("model", "q4_k_m", "llamacpp", "cpu", 0)
    path = tmp_path / "bench.json"
    save_cache({key: benchmark(lambda _prefill, _decode: 3.0)}, path)
    assert load_cache(path)[key] == 3.0


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


def test_supervisor_dry_run_contains_argv(tmp_path, catalog: list[object]) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    result = Supervisor(state_path=tmp_path / "state.json").up(plan, dry_run=True)
    assert result.services
    assert result.services[0]["argv"] == plan.services[0].launch.argv


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
    supervisor.down()
    assert process.poll() is None


def test_supervisor_status_state_fallback_is_running(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"services": [{"service": "chat", "pid": 123, "port": 18010}]}',
        encoding="utf-8",
    )
    result = Supervisor(state_path=state_path).status()
    assert result.running is True
    assert result.services[0]["service"] == "chat"


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
    assert any("再計画しました" in warning for warning in supervisor.active_plan.warnings)
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
