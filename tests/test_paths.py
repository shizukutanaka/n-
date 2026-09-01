from __future__ import annotations

from pathlib import Path

from nmesh.paths import nmesh_home, state_dir


def test_nmesh_home_uses_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path / "custom"))
    assert nmesh_home() == tmp_path / "custom"
    assert state_dir() == tmp_path / "custom"
