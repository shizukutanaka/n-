"""Shared llama.cpp engine identity for speculation evidence.

Speculation evidence is only valid for the build it was measured on, so the
planner's gate lookup and the measurement CLI have to name the engine the same
way. Both call this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from nmesh.probe import HardwareProfile


def engine_identity(profile: HardwareProfile | None = None) -> str:
    """Name the llama.cpp build that a measurement or a plan is bound to.

    Prefers the engine nmesh installed and can report a version line for, and
    falls back to what the hardware probe found on the machine.
    """
    # Imported here because nmesh.runtime imports the planner, which imports
    # this module.
    from nmesh.runtime.engine import active

    installed = active()
    if installed is not None and installed.version_line:
        return installed.version_line
    if profile is None:
        return ""
    return str(profile.available_backends.get("llamacpp") or "")


__all__ = ["engine_identity"]
