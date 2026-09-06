"""Exact uncertainty checks for the small evaluation suite.

The 16-task suite is useful for falsifying large quality-prior errors, but its
0.05 planner gate is below the exact Fisher resolving power of 0.3125. The
measured 12/16 versus 9/16 comparison had p=0.4578, and the 1-vs-1 paired
discordance had McNemar p=1.0. Model selection remains unchanged; a measured
quality floor is not yet justified at this suite size.
"""

from __future__ import annotations

import math
from functools import lru_cache


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the clamped Wilson confidence interval for a binomial rate."""
    if n <= 0:
        return 0.0, 1.0
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Return the exact two-sided Fisher p-value for a 2x2 table."""
    total = a + b + c + d
    if total <= 0:
        return 1.0
    row_one = a + b
    column_one = a + c
    denominator = math.comb(total, column_one)

    def probability(top_left: int) -> float:
        return (
            math.comb(row_one, top_left)
            * math.comb(total - row_one, column_one - top_left)
            / denominator
        )

    observed = probability(a)
    lower = max(0, row_one - (total - column_one))
    upper = min(row_one, column_one)
    return min(
        1.0,
        sum(
            probability(top_left)
            for top_left in range(lower, upper + 1)
            if probability(top_left) <= observed + 1e-12
        ),
    )


def mcnemar_two_sided(b: int, c: int) -> float:
    """Return the exact two-sided McNemar binomial p-value."""
    trials = b + c
    if trials == 0:
        return 1.0
    observed = math.comb(trials, b) / 2**trials
    return min(
        1.0,
        sum(
            math.comb(trials, successes) / 2**trials
            for successes in range(trials + 1)
            if math.comb(trials, successes) / 2**trials <= observed + 1e-12
        ),
    )


@lru_cache(maxsize=None)  # noqa: UP033
def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Return the minimum discordant count that can reach ``alpha``.

    The measured 104-task comparison had 24 discordances one way and 5 the
    other; adding concordant tasks cannot improve its paired resolving power.
    """
    discordant = 0
    while mcnemar_two_sided(discordant, 0) >= alpha:
        discordant += 1
    return discordant


@lru_cache(maxsize=None)  # noqa: UP033
def min_discordant_imbalance(discordant: int) -> int | None:
    """Return the smallest significant imbalance for a discordant total.

    For the measured pair, the 24/5 split is significant, while a total of
    five discordances cannot reach 0.05 regardless of how it is split.
    """
    if discordant < 0:
        return None
    significant = [
        abs(b - (discordant - b))
        for b in range(discordant + 1)
        if mcnemar_two_sided(b, discordant - b) < 0.05
    ]
    return min(significant) if significant else None


@lru_cache(maxsize=None)  # noqa: UP033
def min_resolvable_difference(n: int) -> float:
    """Return the smallest exact Fisher-resolvable rate gap for ``n`` tasks."""
    if n <= 0:
        return 1.0
    for difference in range(1, n + 1):
        for k in range(n - difference + 1):
            j = k + difference
            if fisher_two_sided(j, n - j, k, n - k) < 0.05:
                return difference / n
    return 1.0
