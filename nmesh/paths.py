from __future__ import annotations

import os
from pathlib import Path


def is_windows() -> bool:
    return os.name == "nt"


def nmesh_home() -> Path:
    configured = os.environ.get("NMESH_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".nmesh"
