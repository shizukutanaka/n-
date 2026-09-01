from __future__ import annotations

import os
from pathlib import Path


def nmesh_home() -> Path:
    configured = os.environ.get("NMESH_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".nmesh"


def state_dir() -> Path:
    return nmesh_home()
