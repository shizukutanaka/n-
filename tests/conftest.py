from __future__ import annotations

import socket

import pytest

from nmesh import telemetry
from nmesh.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _telemetry_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr(telemetry, "_default", Telemetry(tmp_path / "telemetry.json"))


@pytest.fixture(autouse=True)
def _no_reverse_dns(monkeypatch) -> None:
    """http.server's server_bind calls socket.getfqdn, which can block for
    tens of seconds on hosts with broken/slow DNS — test HTTP servers never
    use server_name, so skip the lookup entirely."""
    monkeypatch.setattr(socket, "getfqdn", lambda _name="": "localhost")
