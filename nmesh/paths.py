from __future__ import annotations

import os
from pathlib import Path


def is_windows() -> bool:
    return os.name == "nt"


def nmesh_home() -> Path:
    configured = os.environ.get("NMESH_HOME")
    if not configured:
        return Path.home() / ".nmesh"
    path = Path(configured).expanduser()
    # A relative NMESH_HOME would silently break every path-identity check
    # (engine exes are absolute; recorded state is absolute) — anchor it at
    # the caller's cwd instead of comparing `relative` vs `absolute` paths.
    return path if path.is_absolute() else Path.cwd() / path
