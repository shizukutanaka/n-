"""Shared low-level evidence predicates."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, ClassVar, Protocol, TypeVar

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


class _ReferencedRecord(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def reference_id(self) -> str: ...

    @property
    def reference_tps(self) -> float: ...


_ReferencedT = TypeVar("_ReferencedT", bound=_ReferencedRecord)


def demote_refuted(
    records: dict[str, _ReferencedT],
    reference_id: str,
    reference_tps: float,
) -> tuple[str, ...]:
    """Mark every record measured against ``reference_id`` whose reference
    throughput a faster ``reference_tps`` disproves as ``epoch="degraded"``."""
    demoted: list[str] = []
    for key, record in records.items():
        if (
            not reference_id
            or record.reference_id != reference_id
            or not refutes(record.reference_tps, reference_tps)
        ):
            continue
        records[key] = replace(record, epoch="degraded")
        demoted.append(key)
    return tuple(sorted(demoted))
