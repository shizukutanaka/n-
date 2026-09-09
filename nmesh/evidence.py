"""Shared low-level evidence predicates."""

from __future__ import annotations

import math

EPOCH_MIN_RATIO = 0.80


def refutes(recorded_tps: float, reference_tps: float) -> bool:
    """Return whether a faster reference disproves recorded throughput."""
    return (
        math.isfinite(recorded_tps)
        and recorded_tps > 0
        and math.isfinite(reference_tps)
        and reference_tps > 0
        and reference_tps >= recorded_tps / EPOCH_MIN_RATIO
    )
