"""Persisted delegation evidence and the gate that consumes it.

Delegation costs extra calls on every request, so nmesh refuses to serve it
until a measurement on this machine shows the pair beating the lead model
alone. The record is keyed by the full identity of both roles (model,
quantization, backend, resolved artifact), the suite digest, the reasoning
allowance and the protocol version, so a record never justifies a
configuration it was not measured on.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from nmesh.paths import nmesh_home

from .measure import DelegationRun, RoleIdentity

ALPHA = 0.05

#: Delegation is served.
ALLOW = "allow"
#: No measurement exists for this exact pair and suite.
NO_EVIDENCE = "no_evidence"
#: A measurement exists and does not show delegation ahead of the lead.
NOT_SUPERIOR = "not_superior"


@dataclass(frozen=True)
class DelegationRecord:
    """A stored delegation measurement, without the per-task rows."""

    lead: RoleIdentity
    worker: RoleIdentity
    suite: str
    digest: str
    n_tasks: int
    worker_passed: int
    lead_passed: int
    delegated_passed: int
    ceiling_passed: int
    delegated_gained: int
    delegated_lost: int
    delegated_p: float
    ceiling_gained: int
    ceiling_lost: int
    ceiling_p: float
    verifier_accuracy: float
    accepted_but_wrong: int
    rejected_but_right: int
    lead_tokens_solo: int
    lead_tokens_delegated: int
    seconds_solo: float
    seconds_delegated: float
    unscorable: int
    reasoning_allowance: int
    protocol: str
    at: float

    @property
    def superior(self) -> bool:
        """Whether the measurement puts delegation ahead of the lead alone."""
        return (
            self.delegated_passed > self.lead_passed
            and self.delegated_p < ALPHA
            and self.unscorable == 0
        )

    @property
    def ceiling_resolvable(self) -> bool:
        """Whether a perfect verifier could beat the lead on this suite."""
        return self.ceiling_passed > self.lead_passed and self.ceiling_p < ALPHA


def from_run(run: DelegationRun) -> DelegationRecord:
    """Drop the per-task rows and keep the comparable summary."""
    return DelegationRecord(
        lead=run.lead,
        worker=run.worker,
        suite=run.suite,
        digest=run.digest,
        n_tasks=run.n_tasks,
        worker_passed=run.worker_passed,
        lead_passed=run.lead_passed,
        delegated_passed=run.delegated_passed,
        ceiling_passed=run.ceiling_passed,
        delegated_gained=run.delegated_vs_lead.gained,
        delegated_lost=run.delegated_vs_lead.lost,
        delegated_p=run.delegated_vs_lead.p,
        ceiling_gained=run.ceiling_vs_lead.gained,
        ceiling_lost=run.ceiling_vs_lead.lost,
        ceiling_p=run.ceiling_vs_lead.p,
        verifier_accuracy=run.verifier.accuracy,
        accepted_but_wrong=run.verifier.accepted_but_wrong,
        rejected_but_right=run.verifier.rejected_but_right,
        lead_tokens_solo=run.lead_tokens_solo,
        lead_tokens_delegated=run.lead_tokens_delegated,
        seconds_solo=run.seconds_solo,
        seconds_delegated=run.seconds_delegated,
        unscorable=run.unscorable,
        reasoning_allowance=run.reasoning_allowance,
        protocol=run.protocol,
        at=run.at,
    )


def role_key(role: RoleIdentity) -> str:
    return f"{role.model_id}|{role.quant}|{role.backend}|{role.artifact}"


def delegation_key(
    lead: RoleIdentity,
    worker: RoleIdentity,
    suite: str,
    digest: str,
    protocol: str,
    reasoning_allowance: int = 0,
) -> str:
    """Identify a delegation measurement.

    A different verifier prompt, a different token budget or a different
    resolved artifact is a different measurement, so each of them changes the
    key instead of silently reusing a record.
    """
    suffix = f"|a{reasoning_allowance}" if reasoning_allowance else ""
    return (
        f"{role_key(lead)}>{role_key(worker)}"
        f"|{suite}|{digest}|{protocol}{suffix}"
    )


def record_key(record: DelegationRecord) -> str:
    return delegation_key(
        record.lead,
        record.worker,
        record.suite,
        record.digest,
        record.protocol,
        record.reasoning_allowance,
    )


def decide(record: DelegationRecord | None) -> tuple[str, str]:
    """Return the gate decision and the reason to report with it."""
    if record is None:
        return NO_EVIDENCE, NO_EVIDENCE
    if record.superior:
        return ALLOW, ALLOW
    return NOT_SUPERIOR, NOT_SUPERIOR


def best_for(
    cache: Mapping[str, DelegationRecord],
    lead: RoleIdentity,
    worker: RoleIdentity,
    protocol: str,
) -> DelegationRecord | None:
    matches = [
        item for item in cache.values()
        if (
            item.lead == lead
            and item.worker == worker
            and item.protocol == protocol
            and item.superior
        )
    ]
    return max(matches, key=lambda item: item.at, default=None)


def cache_path(path: Path | None = None) -> Path:
    return path or (nmesh_home() / "delegation.json")


def load_cache(path: Path | None = None) -> dict[str, DelegationRecord]:
    target = cache_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, Mapping):
        return {}
    return {
        str(key): parsed
        for key, value in results.items()
        if (parsed := _parse(value)) is not None
    }


def save(run: DelegationRun, path: Path | None = None) -> Path:
    """Merge one measurement into the cache and return the file written."""
    target = cache_path(path)
    records = load_cache(target)
    record = from_run(run)
    records[record_key(record)] = record
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


def _role(data: object) -> RoleIdentity | None:
    if not isinstance(data, Mapping):
        return None
    model_id = data.get("model_id")
    quant = data.get("quant")
    backend = data.get("backend")
    artifact = data.get("artifact", "")
    if (
        not isinstance(model_id, str)
        or not isinstance(quant, str)
        or not isinstance(backend, str)
        or not isinstance(artifact, str)
    ):
        return None
    return RoleIdentity(model_id, quant, backend, artifact)


def _count(value: object, limit: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    if limit is not None and value > limit:
        return None
    return value


def _ratio(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return float(value)


def _parse(data: object) -> DelegationRecord | None:
    if not isinstance(data, Mapping):
        return None
    lead = _role(data.get("lead"))
    worker = _role(data.get("worker"))
    suite = data.get("suite")
    digest = data.get("digest")
    protocol = data.get("protocol")
    at = data.get("at")
    if (
        lead is None
        or worker is None
        or not isinstance(suite, str)
        or not isinstance(digest, str)
        or not isinstance(protocol, str)
        or isinstance(at, bool)
        or not isinstance(at, (int, float))
        or not math.isfinite(at)
    ):
        return None
    n_tasks = _count(data.get("n_tasks"))
    if n_tasks is None or n_tasks == 0:
        return None
    counts = {
        name: _count(data.get(name), n_tasks)
        for name in (
            "worker_passed",
            "lead_passed",
            "delegated_passed",
            "ceiling_passed",
            "delegated_gained",
            "delegated_lost",
            "ceiling_gained",
            "ceiling_lost",
            "accepted_but_wrong",
            "rejected_but_right",
            "unscorable",
        )
    }
    tokens = {
        name: _count(data.get(name))
        for name in ("lead_tokens_solo", "lead_tokens_delegated")
    }
    allowance = _count(data.get("reasoning_allowance", 0))
    accuracy = _ratio(data.get("verifier_accuracy"))
    delegated_p = _ratio(data.get("delegated_p"))
    ceiling_p = _ratio(data.get("ceiling_p"))
    seconds = {
        name: data.get(name) for name in ("seconds_solo", "seconds_delegated")
    }
    if (
        any(value is None for value in counts.values())
        or any(value is None for value in tokens.values())
        or allowance is None
        or accuracy is None
        or delegated_p is None
        or ceiling_p is None
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in seconds.values()
        )
    ):
        return None
    return DelegationRecord(
        lead=lead,
        worker=worker,
        suite=suite,
        digest=digest,
        n_tasks=n_tasks,
        verifier_accuracy=accuracy,
        delegated_p=delegated_p,
        ceiling_p=ceiling_p,
        reasoning_allowance=allowance,
        protocol=protocol,
        at=float(at),
        seconds_solo=float(seconds["seconds_solo"]),
        seconds_delegated=float(seconds["seconds_delegated"]),
        lead_tokens_solo=tokens["lead_tokens_solo"],
        lead_tokens_delegated=tokens["lead_tokens_delegated"],
        **counts,
    )
