from __future__ import annotations

import socket

import pytest

from nmesh import artifacts, telemetry
from nmesh import gateway as gateway_module
from nmesh.bench import cache as bench_cache
from nmesh.bench import epoch as epoch_module
from nmesh.planner import core as planner_core
from nmesh.probe import caps as caps_module
from nmesh.runtime import supervisor as supervisor_module
from nmesh.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _telemetry_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr(planner_core, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(gateway_module, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(bench_cache, "CACHE_PATH", tmp_path / "bench.json")
    monkeypatch.setattr(epoch_module, "EPOCH_PATH", tmp_path / "epoch.json")
    monkeypatch.setattr(artifacts, "CACHE_PATH", tmp_path / "artifacts.json")
    monkeypatch.setattr(caps_module, "CAPS_PATH", tmp_path / "caps.json")
    monkeypatch.setattr(supervisor_module, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(supervisor_module._default, "state_path", tmp_path / "state.json")
    monkeypatch.setattr(telemetry, "_default", Telemetry(tmp_path / "telemetry.json"))


@pytest.fixture(autouse=True)
def _no_reverse_dns(monkeypatch) -> None:
    """http.server's server_bind calls socket.getfqdn, which can block for
    tens of seconds on hosts with broken/slow DNS — test HTTP servers never
    use server_name, so skip the lookup entirely."""
    monkeypatch.setattr(socket, "getfqdn", lambda _name="": "localhost")
