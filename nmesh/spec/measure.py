"""Falsify a speculative-decoding configuration before a plan enables it.

llama.cpp can decode speculatively either with a draft model
(``--spec-type draft-simple --spec-draft-model``) or from the prompt's own
n-grams (``--spec-type ngram-simple``). Both are sold as free speed: the
target model verifies every drafted token, so greedy output is supposed to be
the same tokens the target would have produced alone, only sooner.

Neither half of that claim survives contact with a machine. Verification of
``k`` drafted tokens costs roughly ``k`` times the compute of one decode step,
so on a CPU host - where decoding is compute-bound rather than
memory-bandwidth-bound - a high acceptance rate does not imply a speedup. And
batched verification does not reproduce the target's own arithmetic on every
build, so the output is not always identical.

This module therefore measures one *arm* (one running server, one spec
configuration) over a fixed set of workload classes and lets a caller compare
arms. The classes exist because the payoff is class-dependent: an answer that
copies from its prompt is drafted almost perfectly, while free prose is not.

Sampling is pinned (temperature 0, ``top_k`` 1, fixed seed, prompt cache off)
and the content of every run is hashed, so "the same tokens" is a measured
fact rather than an assumption.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import httpx

from nmesh.orchestrate.measure import RoleIdentity

#: Bumped when the workloads or the sampling settings change, so a record
#: never justifies a configuration measured under different conditions.
SPEC_HARNESS_VERSION = "spec-v1"

#: Speculation off: the reference arm every candidate is compared against.
KIND_NONE = "none"
#: Draft tokens come from a second, smaller model held in memory.
KIND_DRAFT = "draft"
#: Draft tokens come from n-grams of the prompt; no extra weights.
KIND_NGRAM = "ngram"

KINDS = (KIND_NONE, KIND_DRAFT, KIND_NGRAM)

_COPY_SOURCE = (
    "The nmesh planner probes the machine, estimates the memory each artifact "
    "needs, picks a quantization, and then launches a backend process. It "
    "records the real byte count of every artifact it downloads so later "
    "plans stop trusting the nominal bits-per-weight table. When free memory "
    "is lower than the plan assumed, admission re-plans before launching "
    "instead of failing at the health check."
)


@dataclass(frozen=True)
class Workload:
    """One workload class, fixed so two arms are comparable."""

    name: str
    prompt: str
    max_tokens: int


#: Four classes chosen for how much of the answer a drafter can predict:
#: ``copy`` repeats its own prompt (the best case for n-gram drafting),
#: ``structured`` follows a rigid pattern, ``code`` is partly predictable, and
#: ``prose`` is the worst case.
WORKLOADS: tuple[Workload, ...] = (
    Workload(
        "code",
        "Write a Python function `def flatten(items):` that flattens an "
        "arbitrarily nested list of integers iteratively, with a docstring "
        "and three doctest examples. Output code only.",
        256,
    ),
    Workload(
        "structured",
        "Output a JSON array of the integers 1 through 40, each wrapped as "
        '{"n": <int>, "square": <int>}. Output JSON only.',
        256,
    ),
    Workload(
        "prose",
        "Write one paragraph about why measuring a system is harder than "
        "building it.",
        256,
    ),
    Workload(
        "copy",
        "Repeat the following text back verbatim, then list its three main "
        f"claims as a numbered list.\n\n{_COPY_SOURCE}",
        256,
    ),
)


@dataclass(frozen=True)
class SpecConfig:
    """The speculative configuration an arm was measured with."""

    kind: str = KIND_NONE
    draft: RoleIdentity | None = None
    n_max: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown speculation kind: {self.kind}")
        if (self.kind == KIND_DRAFT) != (self.draft is not None):
            raise ValueError(
                "A draft identity is required exactly for kind 'draft'"
            )


@dataclass(frozen=True)
class ClassResult:
    """What one workload class did on one arm."""

    name: str
    completion_tokens: int
    #: Worst decode rate seen across the repeats, in tokens per second.
    decode_tps: float
    #: Wall time of the fastest repeat, in seconds.
    seconds: float
    #: SHA-256 of the answer text, so identity is comparable without storing
    #: the answers themselves.
    content_sha256: str
    #: Whether the repeats disagreed with each other. An unstable arm cannot
    #: support any claim about identity or speed.
    unstable: bool
    #: Tokens drafted and accepted, when the backend reports them.
    drafted: int = 0
    accepted: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


@dataclass(frozen=True)
class ArmRun:
    """One server, one spec configuration, all workload classes."""

    target: RoleIdentity
    spec: SpecConfig
    classes: tuple[ClassResult, ...]
    repeats: int
    harness: str
    at: float

    def by_name(self, name: str) -> ClassResult | None:
        return next(
            (item for item in self.classes if item.name == name), None
        )

    @property
    def unstable(self) -> bool:
        return any(item.unstable for item in self.classes)


@dataclass(frozen=True)
class ClassComparison:
    """One workload class, candidate arm against the reference arm."""

    name: str
    #: Worst-case ratio: the candidate's slowest repeat over the reference's
    #: fastest one. A ratio above one survived every repeat.
    speedup: float
    identical: bool
    reference_tps: float
    candidate_tps: float
    acceptance: float


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ask(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    workload: Workload,
) -> tuple[str, int, float, float, int, int]:
    body = {
        "model": model_ref,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": workload.max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,
    }
    started = time.monotonic()
    response = client.post(f"{base_url}/v1/chat/completions", json=body)
    response.raise_for_status()
    elapsed = time.monotonic() - started
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Upstream returned a non-object response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Upstream returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise TypeError("Upstream returned no message content")
    usage = payload.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else 0
    timings = payload.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    reported = timings.get("predicted_per_second")
    completion = int(tokens) if isinstance(tokens, int) else 0
    tps = (
        float(reported)
        if isinstance(reported, (int, float)) and reported > 0
        else (completion / elapsed if elapsed > 0 else 0.0)
    )
    drafted = timings.get("draft_n")
    accepted = timings.get("draft_n_accepted")
    return (
        content,
        completion,
        tps,
        elapsed,
        int(drafted) if isinstance(drafted, int) else 0,
        int(accepted) if isinstance(accepted, int) else 0,
    )


def run_arm(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    *,
    target: RoleIdentity,
    spec: SpecConfig,
    repeats: int = 2,
    workloads: tuple[Workload, ...] = WORKLOADS,
) -> ArmRun:
    """Measure every workload class against one already-running server."""
    if repeats < 2:
        raise ValueError("A single repeat cannot show whether an arm is stable")
    results: list[ClassResult] = []
    for workload in workloads:
        answers: list[str] = []
        rates: list[float] = []
        seconds: list[float] = []
        tokens = 0
        drafted = 0
        accepted = 0
        for _ in range(repeats):
            content, completion, tps, elapsed, n_draft, n_ok = _ask(
                client, base_url, model_ref, workload
            )
            answers.append(content)
            rates.append(tps)
            seconds.append(elapsed)
            tokens = completion
            drafted = n_draft
            accepted = n_ok
        results.append(
            ClassResult(
                name=workload.name,
                completion_tokens=tokens,
                decode_tps=min(rates),
                seconds=min(seconds),
                content_sha256=_hash(answers[0]),
                unstable=len(set(answers)) > 1,
                drafted=drafted,
                accepted=accepted,
            )
        )
    return ArmRun(
        target=target,
        spec=spec,
        classes=tuple(results),
        repeats=repeats,
        harness=SPEC_HARNESS_VERSION,
        at=time.time(),
    )


def compare(reference: ArmRun, candidate: ArmRun) -> tuple[ClassComparison, ...]:
    """Compare a candidate arm against speculation-off on the same target."""
    if reference.spec.kind != KIND_NONE:
        raise ValueError("The reference arm must have speculation disabled")
    if reference.target != candidate.target:
        raise ValueError("Arms measured on different targets are not comparable")
    if reference.harness != candidate.harness:
        raise ValueError("Arms measured by different harnesses are not comparable")
    comparisons: list[ClassComparison] = []
    for base in reference.classes:
        other = candidate.by_name(base.name)
        if other is None:
            continue
        comparisons.append(
            ClassComparison(
                name=base.name,
                speedup=(
                    other.decode_tps / base.decode_tps
                    if base.decode_tps > 0 else 0.0
                ),
                identical=(
                    other.content_sha256 == base.content_sha256
                    and not other.unstable
                    and not base.unstable
                ),
                reference_tps=base.decode_tps,
                candidate_tps=other.decode_tps,
                acceptance=other.acceptance,
            )
        )
    return tuple(comparisons)


__all__ = [
    "KINDS",
    "KIND_DRAFT",
    "KIND_NGRAM",
    "KIND_NONE",
    "SPEC_HARNESS_VERSION",
    "WORKLOADS",
    "ArmRun",
    "ClassComparison",
    "ClassResult",
    "SpecConfig",
    "Workload",
    "compare",
    "run_arm",
]
