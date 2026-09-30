from __future__ import annotations

from pathlib import Path

import pytest

from nmesh.paths import nmesh_home


def test_nmesh_home_uses_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path / "custom"))
    assert nmesh_home() == tmp_path / "custom"


def test_telemetry_path_follows_late_nmesh_home(monkeypatch, tmp_path: Path) -> None:
    """The module-level _default Telemetry must not pin NMESH_HOME at import."""
    from nmesh import telemetry

    later = tmp_path / "later"
    monkeypatch.setenv("NMESH_HOME", str(later))
    assert telemetry.Telemetry().path == later / "telemetry.json"
    explicit = tmp_path / "explicit.json"
    assert telemetry.Telemetry(explicit).path == explicit


def test_state_files_follow_late_nmesh_home(monkeypatch, tmp_path: Path) -> None:
    """Module-level path constants must not pin NMESH_HOME at import time."""
    from nmesh.bench import cache as bench_cache
    from nmesh.probe.caps import llamacpp_caps
    from nmesh.runtime.supervisor import Supervisor

    later = tmp_path / "later"
    monkeypatch.setenv("NMESH_HOME", str(later))

    assert Supervisor().state_path == later / "state.json"
    records = bench_cache.save_records({})
    assert records == later / "bench.json"
    assert records.exists()

    binary = tmp_path / "fake-llama-server"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    # cache_path falls back to NMESH_HOME/caps.json when not overridden.
    llamacpp_caps(str(binary))
    assert not (tmp_path / "caps.json").exists()


def test_gateway_plan_path_follows_late_nmesh_home(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import nmesh.gateway as gateway_module

    later = tmp_path / "later"
    monkeypatch.setenv("NMESH_HOME", str(later))
    # _plan_stamp must read the post-import NMESH_HOME, not the import-time one.
    proxy = gateway_module._PlanState.__new__(gateway_module._PlanState)
    assert proxy._plan_stamp() is None
    later.mkdir()
    (later / "plan.json").write_text("{}")
    assert proxy._plan_stamp() is not None
