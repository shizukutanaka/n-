"""Deterministic mention extraction; mentions are not evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .sources import SourceItem


@dataclass(frozen=True)
class Mention:
    kind: str
    value: str
    count: int
    sources: tuple[str, ...]


_FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]{2,}")
_HF_RE = re.compile(
    r"(?<![\w.-])(?:https?://)?(?:www\.)?huggingface\.co/"
    r"([A-Za-z0-9][\w.-]{1,40}/[\w.-]{2,60})(?![\w.-])"
)
_QUANT_RE = re.compile(r"\b(?:[I]?Q\d(?:_[A-Z0-9]+)+|fp16|bf16)\b")
_ROUTE_RE = re.compile(r"/v1/[a-z][a-z0-9./_-]*")


def _matches(kind: str, body: str) -> tuple[str, ...]:
    if kind == "flag":
        return tuple(_FLAG_RE.findall(body))
    if kind == "model_repo":
        return tuple(
            repo for repo in _HF_RE.findall(body)
            if repo.lower() not in {"docs/hub", "blog/nvidia"}
        )
    if kind == "quant":
        return tuple(_QUANT_RE.findall(body))
    return tuple(_ROUTE_RE.findall(body))


def extract(items: Sequence[SourceItem]) -> tuple[Mention, ...]:
    """Extract and aggregate reproducible mentions from source item bodies."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    urls: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        for kind in ("flag", "model_repo", "quant", "route"):
            for value in _matches(kind, item.body):
                key = (kind, value)
                counts[key] += 1
                if item.url not in urls[key] and len(urls[key]) < 5:
                    urls[key].append(item.url)
    mentions = [
        Mention(kind, value, count, tuple(urls[(kind, value)]))
        for (kind, value), count in counts.items()
    ]
    return tuple(sorted(mentions, key=lambda mention: (-mention.count, mention.value)))


__all__ = ["Mention", "extract"]
