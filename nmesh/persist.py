"""Shared reader and cache writer for nmesh-owned JSON state files.

A missing state file is a normal first-run condition and stays silent. A
file that exists but cannot be parsed — corrupt, hand-edited into invalid
JSON, or unreadable — would otherwise be silently discarded and replaced
with a default, so it warns once per path per process.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

from nmesh import i18n

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_warned: set[str] = set()


def int_at_least(value: object, minimum: int) -> TypeGuard[int]:
    """True for a non-bool int that is at least *minimum*."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


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


def finite_number(value: object) -> TypeGuard[float]:
    """True for a non-bool int or float that is finite."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def staging_path(target: Path) -> Path:
    """Per-process temporary file beside *target* used for atomic replaces."""
    return target.with_name(f".{target.name}.{os.getpid()}.tmp")


def write_json_atomic(
    target: Path, payload: object, *, ensure_ascii: bool = True,
) -> None:
    """Write *payload* as indented JSON to *target* via a staged temporary
    file and ``os.replace`` so a crash mid-write never leaves a truncated
    file. Errors propagate; the staging file is left for the caller."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_path(target)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def write_results_cache(
    target: Path, records: Mapping[str, DataclassInstance],
) -> None:
    """Atomically replace *target* with ``{"results": {key: record}}``.

    The payload is staged in a per-process temporary file beside *target*
    so a crash mid-write never leaves a truncated cache; a failed write
    removes its own staging file and re-raises.
    """
    try:
        write_json_atomic(
            target,
            {"results": {key: asdict(value) for key, value in records.items()}},
            ensure_ascii=False,
        )
    except OSError:
        try:
            staging_path(target).unlink()
        except OSError:
            pass
        raise
