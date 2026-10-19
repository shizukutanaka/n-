"""Shared low-level evidence predicates."""

from __future__ import annotations

import math
from typing import Protocol

EPOCH_MIN_RATIO = 0.80


class _Identified(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def quant(self) -> str: ...

    @property
    def backend(self) -> str: ...


def identity_key(record: _Identified) -> tuple[str, str, str]:
    """Case-insensitive (model, quant, backend) key shared by evidence lookups."""
    return (
        record.model_id.casefold(),
        record.quant.casefold(),
        record.backend.casefold(),
    )


def refutes(recorded_tps: float, reference_tps: float) -> bool:
    """Return whether a faster reference disproves recorded throughput."""
    return (
        math.isfinite(recorded_tps)
        and recorded_tps > 0
        and math.isfinite(reference_tps)
        and reference_tps > 0
        and reference_tps >= recorded_tps / EPOCH_MIN_RATIO
    )
