"""Deterministic context-depth probes for quality evidence.

The existing hard suite cannot license a context depth: 40/40 native tasks
fell to 38/40 at about 1,220 real prompt tokens, 39/40 at 4,978, 39/40 at
10,007, and 37/40 at 15,033, while shallow controls stayed 40/40. Context
retrieval probes are therefore separate. Literal and latent single-needle
tasks held 80/80 at about 110, 1,245, 5,020, 10,030, and 15,050 real tokens,
with needles at both 10% and 90% of the filler. On this host, those depths
cost about 0.4, 3.1, 14.7, 35.7, and 63.0 seconds respectively, or about
242 prefill tokens per second in the quiet run. These measurements justify
recording depth coverage and warning about evidence gaps, not excluding a
context that fits the KV cache.

The paired probes also include four scattered multi-code tasks: they scored
7/8 natively, then 4/4 at about 1,200 tokens, 3/4 at 5,000, 4/4 at 10,000,
and 3/4 at 15,000. Aggregate tasks scored 4/8 natively and 1/4, 2/4, 0/4,
and 2/4 deeply; order tasks scored 2/8 natively and 1/4, 1/4, 2/4, and 1/4
deeply. Aggregate and order therefore fail at about 140 tokens on this
artifact, so a family whose shallow control fails cannot be read as a depth
result.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Sequence

from .suite import Task

_FILLER_NOUNS = (
    "archive", "beacon", "canvas", "detail", "engine", "folder", "ledger",
    "packet", "record", "signal", "status", "thread", "window", "workbench",
)
_FILLER_ADJECTIVES = (
    "amber", "careful", "distant", "even", "fresh", "gentle", "plain",
    "quiet", "steady", "useful", "violet", "wooden",
)
_FILLER_VERBS = (
    "crosses", "follows", "keeps", "marks", "moves", "records", "rests",
    "tracks", "updates", "waits",
)
_CITIES = (
    ("Osaka", "Japan"),
    ("Lyon", "France"),
    ("Bergen", "Norway"),
    ("Cusco", "Peru"),
    ("Perth", "Australia"),
    ("Split", "Croatia"),
)
_NAMES = ("Marta", "Devrim", "Ines", "Kwame", "Lena", "Tomas", "Sanne", "Rafal")
_HEX_RE = re.compile(r"[0-9a-f]{6}")


def _sentence(rng: random.Random) -> str:
    return (
        f"The {rng.choice(_FILLER_ADJECTIVES)} {rng.choice(_FILLER_NOUNS)} "
        f"{rng.choice(_FILLER_VERBS)} the {rng.choice(_FILLER_NOUNS)}."
    )


def _estimate_tokens(text: str) -> int:
    # A top-level import would cycle through gateway, orchestrate, and eval.
    from nmesh.gateway.tokens import estimate_tokens

    return estimate_tokens(text)


def _filler(rng: random.Random, minimum_tokens: int, prefix: str) -> list[str]:
    sentences: list[str] = []
    while _estimate_tokens(f"{prefix}{' '.join(sentences)}") < minimum_tokens:
        sentences.append(_sentence(rng))
    return sentences


def padded_prompt(prompt: str, target: int, seed: str) -> str:
    """Add deterministic irrelevant reference material before the question."""
    if target <= 0:
        return prompt
    case = uuid.uuid5(uuid.NAMESPACE_URL, seed)
    prefix = (
        f"Reference material (case {case}, may be irrelevant):\n"
    )
    suffix = "\nEnd of reference material.\n\nAnswer the question below.\n\n"
    rng = random.Random(seed)
    sentences = _filler(rng, target, prefix + suffix + prompt)
    return prefix + " ".join(sentences) + suffix + prompt


def _needle_prompt(
    question: str,
    needle: str,
    target: int,
    seed: str,
    position: float,
) -> str:
    prefix = "Reference material:\n"
    suffix = "\nEnd of reference material.\n\nAnswer the question below.\n\n"
    rng = random.Random(seed)
    sentences = _filler(rng, target, prefix + suffix + needle + question)
    index = round(len(sentences) * position)
    index = max(0, min(index, len(sentences)))
    sentences.insert(index, needle)
    return prefix + " ".join(sentences) + suffix + question


def _scattered_prompt(
    question: str,
    needles: Sequence[str],
    target: int,
    seed: str,
) -> str:
    prefix = "Reference material:\n"
    suffix = "\nEnd of reference material.\n\nAnswer the question below.\n\n"
    rng = random.Random(seed)
    sentences = _filler(
        rng,
        target,
        prefix + suffix + " ".join(needles) + question,
    )
    for index, needle in enumerate(needles):
        position = round(len(sentences) * (index + 1) / (len(needles) + 1))
        sentences.insert(position + index, needle)
    return prefix + " ".join(sentences) + suffix + question


def _literal_task(target: int, seed: str, index: int, position: float) -> Task:
    case = uuid.uuid5(uuid.NAMESPACE_URL, f"{seed}|literal|{index}|{position}")
    case_id = case.hex[:8]
    code = case.hex[8:14]
    needle = f"The access code for case {case_id} is {code}."
    question = f"What is the access code for case {case_id}? Answer the code only."

    def check(text: str) -> bool:
        match = _HEX_RE.search(text.casefold())
        return match is not None and match.group() == code

    return Task(
        f"context.literal.p{round(position * 100)}.{index}",
        "context.literal",
        _needle_prompt(
            question, needle, target, f"{seed}|literal|{index}|{position}", position,
        ) if target > 0 else f"{needle}\n\n{question}",
        32,
        check,
        grades="value",
    )


def _latent_task(target: int, seed: str, index: int, position: float) -> Task:
    city, country = _CITIES[index % len(_CITIES)]
    name = _NAMES[index]
    needle = f"{name} spent the whole quarter working out of {city}."
    question = f"Which person spent the quarter in {country}? Answer with the name only."

    def check(text: str) -> bool:
        if not text.split():
            return False
        return text.split()[0].strip("`\"' .!,。") == name

    return Task(
        f"context.latent.p{round(position * 100)}.{index}",
        "context.latent",
        _needle_prompt(
            question, needle, target, f"{seed}|latent|{index}|{position}", position,
        ) if target > 0 else f"{needle}\n\n{question}",
        32,
        check,
        grades="value",
    )


def _multi_task(target: int, seed: str, index: int) -> Task:
    cases = tuple(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{seed}|multi|{index}|{number}")
        for number in range(4)
    )
    case_ids = tuple(case.hex[:8] for case in cases)
    codes = [case.hex[8:14] for case in cases]
    needles = tuple(
        f"The access code for case {case_id} is {code}."
        for case_id, code in zip(case_ids, codes, strict=True)
    )
    question = (
        "List the access codes for cases "
        + ", ".join(case_ids)
        + " in exactly that order, separated by commas. "
        "Answer with the four codes only."
    )

    def check(text: str) -> bool:
        return _HEX_RE.findall(text.casefold())[:4] == codes

    prompt = (
        _scattered_prompt(
            question,
            needles,
            target,
            f"{seed}|multi|{index}",
        )
        if target > 0
        else "\n".join(needles) + f"\n\n{question}"
    )
    return Task(
        f"context.multi.{index}",
        "context.multi",
        prompt,
        64,
        check,
        grades="value",
    )


def needle_tasks(target: int, seed: str) -> tuple[Task, ...]:
    """Return paired literal, latent, and scattered multi-code probes."""
    positions = (0.1, 0.9)
    tasks = tuple(
        task
        for position in positions
        for index in range(4)
        for task in (
            _literal_task(target, seed, index, position),
            _latent_task(target, seed, index, position),
        )
    )
    return tasks + tuple(_multi_task(target, seed, index) for index in range(4))


__all__ = ["needle_tasks", "padded_prompt"]
