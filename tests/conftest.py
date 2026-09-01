from __future__ import annotations

import pytest

from nmesh import telemetry
from nmesh.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _telemetry_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "_default", Telemetry(tmp_path / "telemetry.json"))
