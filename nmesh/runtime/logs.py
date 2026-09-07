from __future__ import annotations

import os
import re
from pathlib import Path
from typing import BinaryIO

from nmesh.paths import nmesh_home

LOG_MAX_BYTES = 5 * 1024 * 1024
_TAIL_BYTES = 256 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe(name: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(name)) and name not in {".", ".."}


def _max_bytes() -> int:
    return int(os.environ.get("NMESH_LOG_MAX_BYTES", LOG_MAX_BYTES))


def log_dir() -> Path:
    return nmesh_home() / "logs"


def log_path(name: str) -> Path:
    if not _safe(name):
        raise ValueError(f"unsafe log name: {name}")
    return log_dir() / f"{name}.log"


def rotate(path: Path) -> None:
    if path.exists() and path.stat().st_size >= _max_bytes():
        os.replace(path, Path(f"{path}.1"))


def open_log(name: str) -> BinaryIO:
    try:
        path = log_path(name)
    except ValueError as error:
        raise OSError(str(error)) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate(path)
    return path.open("ab")


def tail(name: str, lines: int = 20) -> list[str]:
    if lines <= 0:
        return []
    try:
        path = log_path(name)
    except ValueError:
        return []
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _TAIL_BYTES))
        content = handle.read().decode("utf-8", errors="replace")
    non_empty = [line.strip() for line in content.splitlines() if line.strip()]
    return non_empty[-lines:]


def available() -> list[str]:
    directory = log_dir()
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.log") if path.is_file())
