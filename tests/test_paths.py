from __future__ import annotations

from pathlib import Path

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
