"""Shared reader for nmesh-owned JSON state files.

A missing state file is a normal first-run condition and stays silent. A
file that exists but cannot be parsed — corrupt, hand-edited into invalid
JSON, or unreadable — would otherwise be silently discarded and replaced
with a default, so it warns once per path per process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nmesh import i18n

_warned: set[str] = set()


def read_json_file(target: Path) -> object | None:
    """Parse *target* as JSON, warning once when it exists but cannot be read."""
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        _warn(target)
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _warn(target)
        return None


def _warn(target: Path) -> None:
    key = str(target)
    if key in _warned:
        return
    _warned.add(key)
    print(
        i18n.t("warn.file_unreadable", i18n.lang(), file=target.name),
        file=sys.stderr,
    )
