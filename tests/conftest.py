from __future__ import annotations

import socket

import pytest

from nmesh import telemetry
from nmesh.planner import core as planner_core
from nmesh.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _telemetry_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr(telemetry, "_default", Telemetry(tmp_path / "telemetry.json"))
    # Ambient loader/linker env (e.g. LD_LIBRARY_PATH on CI runners) is
    # unrelated to the code under test — drop it so plans are deterministic.
    for name in planner_core._RESOLUTION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_reverse_dns(monkeypatch) -> None:
    """http.server's server_bind calls socket.getfqdn, which can block for
    tens of seconds on hosts with broken/slow DNS — test HTTP servers never
    use server_name, so skip the lookup entirely."""
    monkeypatch.setattr(socket, "getfqdn", lambda _name="": "localhost")
