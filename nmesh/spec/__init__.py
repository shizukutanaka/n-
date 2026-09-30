"""Speculative decoding: measured on this machine, off unless it wins.

:mod:`nmesh.spec.measure` measures one running server under one speculation
configuration; :mod:`nmesh.spec.record` stores the comparison against
speculation-off and decides whether a plan may enable it.
"""

from __future__ import annotations

from .engine import engine_identity
from .measure import (
    KIND_DRAFT,
    KIND_NGRAM,
    KIND_NONE,
    KINDS,
    SPEC_HARNESS_VERSION,
    WORKLOADS,
    ArmRun,
    ClassComparison,
    ClassResult,
    ControlCheck,
    SpecConfig,
    Workload,
    compare,
    control,
    request_timeout,
    run_arm,
)
from .record import (
    ALLOW,
    MIN_CONTROL_RATIO,
    MIXED,
    NO_EVIDENCE,
    NOT_FASTER,
    NOT_IDENTICAL,
    STALE,
    UNSTABLE,
    ClassEvidence,
    ControlEvidence,
    SpecRecord,
    best_for,
    decide,
    demote_stale,
    from_arms,
    load_cache,
    save,
    save_all,
    spec_key,
)

__all__ = [
    "ALLOW",
    "KINDS",
    "KIND_DRAFT",
    "KIND_NGRAM",
    "KIND_NONE",
    "MIN_CONTROL_RATIO",
    "MIXED",
    "NOT_FASTER",
    "NOT_IDENTICAL",
    "NO_EVIDENCE",
    "SPEC_HARNESS_VERSION",
    "STALE",
    "UNSTABLE",
    "WORKLOADS",
    "ArmRun",
    "ClassComparison",
    "ClassEvidence",
    "ClassResult",
    "ControlCheck",
    "ControlEvidence",
    "SpecConfig",
    "SpecRecord",
    "Workload",
    "best_for",
    "compare",
    "control",
    "decide",
    "demote_stale",
    "engine_identity",
    "from_arms",
    "load_cache",
    "request_timeout",
    "run_arm",
    "save",
    "save_all",
    "spec_key",
]
