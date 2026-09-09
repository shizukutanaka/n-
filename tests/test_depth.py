from __future__ import annotations

import random
import time

from nmesh.eval import needle_tasks, suite_digest
from nmesh.eval.depth import _estimate_tokens, _filler, _sentence


def _reference_filler(
    rng: random.Random,
    minimum_tokens: int,
    prefix: str,
) -> list[str]:
    sentences: list[str] = []
    while _estimate_tokens(f"{prefix}{' '.join(sentences)}") < minimum_tokens:
        sentences.append(_sentence(rng))
    return sentences


def test_filler_matches_reference_and_rng_state() -> None:
    for target in (0, 1, 37, 256, 1024):
        for seed in ("core", "alpha", "42"):
            prefix = f"Reference material {seed}: "
            expected_rng = random.Random(seed)
            expected = _reference_filler(expected_rng, target, prefix)
            actual_rng = random.Random(seed)
            actual = _filler(actual_rng, target, prefix)
            assert actual == expected
            assert actual_rng.getstate() == expected_rng.getstate()


def test_needle_task_digests_are_stable() -> None:
    assert suite_digest(needle_tasks(1024, "core")) == "v2:c42a1fba335ea00c"
    assert suite_digest(needle_tasks(4096, "core")) == "v2:cc002101bcb7c621"


def test_needle_tasks_8192_is_fast() -> None:
    started = time.perf_counter()
    tasks = needle_tasks(8192, "core")
    elapsed = time.perf_counter() - started
    assert len(tasks) == 24
    # The old implementation took 171.22s; this host takes about 4.6s now.
    assert elapsed < 30


def test_needle_tasks_is_cached() -> None:
    assert needle_tasks(1024, "core") is needle_tasks(1024, "core")
