from __future__ import annotations

from pathlib import Path

from nmesh.paths import nmesh_home


def test_nmesh_home_uses_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path / "custom"))
    assert nmesh_home() == tmp_path / "custom"
