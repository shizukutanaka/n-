"""Persisted speculation evidence and the gate that consumes it.

Speculative decoding is only worth enabling if, on this machine, it produces
the same tokens the target model produces alone *and* produces them faster.
Both halves are measured (see :mod:`nmesh.spec.measure`), and neither is
assumed: a draft model can reach a high acceptance rate and still lose on a
CPU host, and batched verification is not always token-identical.

A record is keyed by the target's full identity, the speculation
configuration, the engine build and the harness version, because every one of
those changes what was measured. Speculation stays off unless a record for
that exact combination allows it.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from nmesh.evidence import refutes
from nmesh.orchestrate.measure import RoleIdentity
from nmesh.orchestrate.record import role_key
from nmesh.paths import nmesh_home

from .measure import (
    KIND_NONE,
    SPEC_HARNESS_VERSION,
    ArmRun,
    ClassComparison,
    SpecConfig,
    compare,
    control,
)

#: A class must be at least this much faster to count as a gain.
MIN_SPEEDUP = 1.05
#: A class slower than this counts as a regression.
MAX_REGRESSION = 0.98

#: Speculation is enabled for this configuration.
ALLOW = "allow"
#: Nothing was measured for this exact configuration.
NO_EVIDENCE = "no_evidence"
#: Speculation changed the output, so it is not free speed.
NOT_IDENTICAL = "not_identical"
#: No workload class gained enough to pay for the arrangement.
NOT_FASTER = "not_faster"
#: Some classes gained and others regressed; nmesh cannot tell which class a
#: request belongs to, so it refuses to bet on the traffic mix.
MIXED = "mixed"
#: The repeats disagreed with each other, so nothing was established.
UNSTABLE = "unstable"
# The host epoch invalidated a positive speed claim.
STALE = "stale"

# The quiet-machine cross-launch spread was 3.3--5.3%; a 10% threshold is
# roughly twice that noise floor and far below the 1.6--2.8x contamination
# that motivated this control.
MIN_CONTROL_RATIO = 0.90


@dataclass(frozen=True)
class ClassEvidence:
    """One workload class of a stored comparison."""

    name: str
    speedup: float
    identical: bool
    reference_tps: float
    candidate_tps: float
    acceptance: float
    reference_spread: float = 0.0
    candidate_spread: float = 0.0


@dataclass(frozen=True)
class ControlEvidence:
    """One workload class of the stored A/A control."""

    name: str
    ratio: float
    identical: bool


@dataclass(frozen=True)
class SpecRecord:
    """One measured speculation configuration on one target."""

    target: RoleIdentity
    spec: SpecConfig
    engine: str
    harness: str
    repeats: int
    classes: tuple[ClassEvidence, ...]
    control: tuple[ControlEvidence, ...] = ()
    reference_id: str = ""
    reference_tps: float = 0.0
    epoch: str = "unknown"
    at: float = 0.0

    @property
    def gains(self) -> tuple[ClassEvidence, ...]:
        return tuple(
            item for item in self.classes if item.speedup >= MIN_SPEEDUP
        )

    @property
    def regressions(self) -> tuple[ClassEvidence, ...]:
        return tuple(
            item for item in self.classes if item.speedup <= MAX_REGRESSION
        )

    @property
    def decision(self) -> str:
        if not self.classes or not self.control:
            return NO_EVIDENCE if not self.classes else UNSTABLE
        if any(
            item.ratio < MIN_CONTROL_RATIO or not item.identical
            for item in self.control
        ):
            return UNSTABLE
        if any(not item.identical for item in self.classes):
            return NOT_IDENTICAL
        if not self.gains:
            return NOT_FASTER
        if self.regressions:
            return MIXED
        # A degraded host can invalidate only a positive speed claim:
        # identity and regression rejections remain valid in any host state.
        if self.epoch == "degraded":
            return STALE
        return ALLOW


def from_arms(
    reference: ArmRun,
    candidate: ArmRun,
    *,
    engine: str,
    control_arm: ArmRun,
    reference_id: str = "",
    reference_tps: float = 0.0,
    epoch: str = "unknown",
) -> SpecRecord:
    """Turn a reference/candidate pair into a storable record."""
    comparisons: Sequence[ClassComparison] = compare(reference, candidate)
    return SpecRecord(
        target=candidate.target,
        spec=candidate.spec,
        engine=engine,
        harness=candidate.harness,
        repeats=min(reference.repeats, candidate.repeats, control_arm.repeats),
        classes=tuple(
            ClassEvidence(
                name=item.name,
                speedup=item.speedup,
                identical=item.identical,
                reference_tps=item.reference_tps,
                candidate_tps=item.candidate_tps,
                acceptance=item.acceptance,
                reference_spread=item.reference_spread,
                candidate_spread=item.candidate_spread,
            )
            for item in comparisons
        ),
        control=tuple(
            ControlEvidence(
                name=item.name,
                ratio=item.ratio,
                identical=item.identical,
            )
            for item in control(reference, control_arm)
        ),
        reference_id=reference_id,
        reference_tps=reference_tps,
        epoch=epoch,
        at=max(reference.at, candidate.at, control_arm.at),
    )


def spec_key(spec: SpecConfig) -> str:
    draft = role_key(spec.draft) if spec.draft is not None else "-"
    return f"{spec.kind}|{draft}|n{spec.n_max}"


def record_key(record: SpecRecord) -> str:
    return (
        f"{role_key(record.target)}>{spec_key(record.spec)}"
        f"|{record.engine}|{record.harness}"
    )


def decide(record: SpecRecord | None) -> tuple[str, str]:
    """Return the gate decision and the reason to report with it."""
    if record is None:
        return NO_EVIDENCE, NO_EVIDENCE
    if any(item.speedup <= 0 for item in record.classes):
        return UNSTABLE, UNSTABLE
    reason = record.decision
    return (ALLOW if reason == ALLOW else reason), reason


def best_for(
    cache: Mapping[str, SpecRecord],
    target: RoleIdentity,
    spec: SpecConfig,
    engine: str,
) -> SpecRecord | None:
    """The newest record for this exact configuration, winning or losing.

    Losing records are kept and returned so a plan can say *why* speculation
    is off instead of reporting that nothing was measured.
    """
    if spec.kind == KIND_NONE:
        return None
    matches = [
        item for item in cache.values()
        if (
            item.target == target
            and item.spec == spec
            and item.engine == engine
            and item.harness == SPEC_HARNESS_VERSION
        )
    ]
    return max(matches, key=lambda item: item.at, default=None)


def cache_path(path: Path | None = None) -> Path:
    return path or (nmesh_home() / "spec.json")


# Loaded twice per `nmesh bench` run (demote+save, then re-read for the
# report) — parse only when (mtime_ns, size) changed.
_cache_memo: dict[Path, tuple[int, int, dict[str, SpecRecord]]] = {}
_cache_lock = threading.Lock()


def load_cache(path: Path | None = None) -> dict[str, SpecRecord]:
    target = cache_path(path)
    try:
        info = target.stat()
        stamp = (info.st_mtime_ns, info.st_size)
    except OSError:
        stamp = (-1, 0)
    with _cache_lock:
        cached = _cache_memo.get(target)
        if cached is not None and cached[0] == stamp[0] and cached[1] == stamp[1]:
            return dict(cached[2])
    records: dict[str, SpecRecord] = {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        payload = None
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if isinstance(results, Mapping):
        records = {
            str(key): parsed
            for key, value in results.items()
            if (parsed := _parse(value)) is not None
        }
    with _cache_lock:
        _cache_memo[target] = (stamp[0], stamp[1], records)
    return dict(records)


def save(record: SpecRecord, path: Path | None = None) -> Path:
    """Merge one record into the cache and return the file written."""
    target = cache_path(path)
    records = load_cache(target)
    records[record_key(record)] = record
    return save_all(records, target)


def save_all(
    records: Mapping[str, SpecRecord],
    path: Path | None = None,
) -> Path:
    """Atomically persist the complete speculation cache."""
    target = cache_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "results": {
                        key: asdict(value) for key, value in records.items()
                    }
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def demote_stale(
    records: dict[str, SpecRecord],
    reference_id: str,
    reference_tps: float,
) -> tuple[str, ...]:
    """Invalidate positive speed claims disproved by a faster epoch."""
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


def _identity(data: object) -> RoleIdentity | None:
    if not isinstance(data, Mapping):
        return None
    values = [data.get(name) for name in ("model_id", "quant", "backend")]
    artifact = data.get("artifact", "")
    if not all(isinstance(item, str) for item in values):
        return None
    if not isinstance(artifact, str):
        return None
    model_id, quant, backend = (str(item) for item in values)
    return RoleIdentity(model_id, quant, backend, artifact)


def _number(value: object, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < minimum:
        return None
    return float(value)


def _spec(data: object) -> SpecConfig | None:
    if not isinstance(data, Mapping):
        return None
    kind = data.get("kind")
    n_max = data.get("n_max", 0)
    if not isinstance(kind, str) or isinstance(n_max, bool):
        return None
    if not isinstance(n_max, int) or n_max < 0:
        return None
    draft_data = data.get("draft")
    draft = _identity(draft_data) if draft_data is not None else None
    if draft_data is not None and draft is None:
        return None
    try:
        return SpecConfig(kind=kind, draft=draft, n_max=n_max)
    except ValueError:
        return None


def _class(data: object) -> ClassEvidence | None:
    if not isinstance(data, Mapping):
        return None
    name = data.get("name")
    identical = data.get("identical")
    if not isinstance(name, str) or not isinstance(identical, bool):
        return None
    numbers = {
        field: _number(data.get(field))
        for field in ("speedup", "reference_tps", "candidate_tps", "acceptance")
    }
    spreads = {
        field: (
            _number(data.get(field))
            if field in data else 0.0
        )
        for field in ("reference_spread", "candidate_spread")
    }
    if (
        any(value is None for value in numbers.values())
        or any(value is None for value in spreads.values())
    ):
        return None
    return ClassEvidence(
        name=name,
        identical=identical,
        speedup=float(numbers["speedup"] or 0.0),
        reference_tps=float(numbers["reference_tps"] or 0.0),
        candidate_tps=float(numbers["candidate_tps"] or 0.0),
        acceptance=float(numbers["acceptance"] or 0.0),
        reference_spread=float(spreads["reference_spread"] or 0.0),
        candidate_spread=float(spreads["candidate_spread"] or 0.0),
    )


def _control(data: object) -> ControlEvidence | None:
    if not isinstance(data, Mapping):
        return None
    name = data.get("name")
    identical = data.get("identical")
    ratio = _number(data.get("ratio"))
    if (
        not isinstance(name, str)
        or not isinstance(identical, bool)
        or ratio is None
    ):
        return None
    return ControlEvidence(name=name, ratio=ratio, identical=identical)


def _parse(data: object) -> SpecRecord | None:
    if not isinstance(data, Mapping):
        return None
    target = _identity(data.get("target"))
    spec = _spec(data.get("spec"))
    engine = data.get("engine")
    harness = data.get("harness")
    repeats = data.get("repeats")
    at = _number(data.get("at"))
    rows = data.get("classes")
    control_rows = data.get("control")
    reference_id = data.get("reference_id", "")
    reference_tps = _number(data.get("reference_tps", 0.0))
    epoch = data.get("epoch", "unknown")
    if (
        target is None
        or spec is None
        or not isinstance(engine, str)
        or not isinstance(harness, str)
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 3
        or at is None
        or not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or (
            control_rows is not None
            and (
                not isinstance(control_rows, Sequence)
                or isinstance(control_rows, (str, bytes))
            )
        )
        or not isinstance(reference_id, str)
        or reference_tps is None
        or not isinstance(epoch, str)
        or epoch not in {"unknown", "healthy", "degraded"}
    ):
        return None
    parsed = [item for row in rows if (item := _class(row)) is not None]
    if len(parsed) != len(rows) or not parsed:
        return None
    parsed_control = (
        [
            control for row in control_rows
            if (control := _control(row)) is not None
        ]
        if control_rows is not None else []
    )
    if (
        control_rows is not None
        and len(parsed_control) != len(control_rows)
    ):
        return None
    return SpecRecord(
        target=target,
        spec=spec,
        engine=engine,
        harness=harness,
        repeats=repeats,
        classes=tuple(parsed),
        control=tuple(parsed_control),
        reference_id=reference_id,
        reference_tps=reference_tps,
        epoch=epoch,
        at=at,
    )


__all__ = [
    "ALLOW",
    "MAX_REGRESSION",
    "MIN_CONTROL_RATIO",
    "MIN_SPEEDUP",
    "MIXED",
    "NOT_FASTER",
    "NOT_IDENTICAL",
    "NO_EVIDENCE",
    "STALE",
    "UNSTABLE",
    "ClassEvidence",
    "ControlEvidence",
    "SpecRecord",
    "best_for",
    "cache_path",
    "decide",
    "demote_stale",
    "from_arms",
    "load_cache",
    "record_key",
    "save",
    "save_all",
    "spec_key",
]
