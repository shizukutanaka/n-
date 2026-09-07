"""Shared llama.cpp engine identity for speculation evidence."""

from __future__ import annotations

from typing import Any


def engine_identity(profile: Any | None = None) -> str:
    """Return the engine identity used by both planning and measurement."""
    from nmesh.runtime.engine import active

    active_engine = active()
    if active_engine is not None and active_engine.version_line:
        return active_engine.version_line
    available = getattr(profile, "available_backends", {}) if profile is not None else {}
    return str(available.get("llamacpp") or "")


__all__ = ["engine_identity"]
