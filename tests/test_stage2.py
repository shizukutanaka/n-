from __future__ import annotations

from dataclasses import replace

import pytest

from nmesh.bench import benchmark, benchmark_key, load_cache, save_cache
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.gateway import route
from nmesh.planner import Policy, build_plan
from nmesh.runtime import Supervisor

from .test_planner import profile


@pytest.fixture
def catalog() -> list[ModelSpec]:
    return load_catalog()


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
    result = supervisor.up(plan, no_download=True)
    assert result.running
    assert len(calls) == 2
    supervisor.down()


def test_supervisor_dry_run_contains_argv(tmp_path, catalog: list[object]) -> None:
    plan = build_plan(profile(8), catalog, Policy(roles=["chat"]))
    result = Supervisor(state_path=tmp_path / "state.json").up(plan, dry_run=True)
    assert result.services
    assert result.services[0]["argv"] == plan.services[0].launch.argv
