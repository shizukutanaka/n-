"""Bounded persistence for watch items and findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nmesh.paths import nmesh_home


@dataclass(frozen=True)
class WatchState:
    last_run: str
    seen_items: dict[str, str]
    seen_findings: dict[str, str]


def _empty() -> WatchState:
    return WatchState("", {}, {})


def _bounded(values: dict[str, str], limit: int = 5000) -> dict[str, str]:
    return dict(sorted(values.items(), key=lambda item: item[1])[-limit:])


def load_state(path: Path | None = None) -> WatchState:
    target = path or (nmesh_home() / "watch.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return _empty()
        items = payload.get("seen_items")
        findings = payload.get("seen_findings")
        if not isinstance(items, dict) or not isinstance(findings, dict):
            return _empty()
        if not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in items.items()):
            return _empty()
        if not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in findings.items()):
            return _empty()
        return WatchState(
            payload.get("last_run", "") if isinstance(payload.get("last_run"), str) else "",
            _bounded(dict(items)),
            _bounded(dict(findings)),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _empty()


def save_state(state: WatchState, path: Path | None = None) -> None:
    target = path or (nmesh_home() / "watch.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "last_run": state.last_run,
        "seen_items": _bounded(state.seen_items),
        "seen_findings": _bounded(state.seen_findings),
    }
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["WatchState", "load_state", "now_iso", "save_state"]
