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
# `caps.json` records the flags of the llama.cpp binaries nmesh launches, so a
# flag is only comparable against it when the article invokes one of them.
# Without this, every `git --no-ext-diff` or `npm --frozen-lockfile` in a post
# became an "unknown flag" finding.
_LLAMACPP_RE = re.compile(
    r"(?<![\w-])llama[-_](?:server|cli|bench|embedding|quantize|perplexity|run)"
    r"(?:\.exe)?(?![\w-])"
)
_CONTINUATION_RE = re.compile(r"\\\s*\n\s*")
_SEGMENT_RE = re.compile(r"\|\||&&|[|;&]")
_PROMPT_RE = re.compile(r"^[$#>\s]+")
_HF_RE = re.compile(
    r"(?<![\w.-])(?:https?://)?(?:www\.)?huggingface\.co/"
    r"([A-Za-z0-9][\w.-]{1,40}/[\w.-]{2,60})(?![\w.-])"
)
_QUANT_RE = re.compile(r"\b(?:[I]?Q\d(?:_[A-Z0-9]+)+|fp16|bf16)\b")
_ROUTE_RE = re.compile(r"/v1/[a-z][a-z0-9./_-]*")


def _flags(body: str) -> tuple[str, ...]:
    """Flags written on a command line that invokes a llama.cpp binary."""
    found: list[str] = []
    for line in _CONTINUATION_RE.sub(" ", body).splitlines():
        for segment in _SEGMENT_RE.split(line):
            segment = _PROMPT_RE.sub("", segment)
            command = segment.split(None, 1)
            if command and _LLAMACPP_RE.search(command[0]):
                found.extend(_FLAG_RE.findall(segment))
    return tuple(found)


def _matches(kind: str, body: str) -> tuple[str, ...]:
    if kind == "flag":
        return _flags(body)
    if kind == "model_repo":
        return tuple(_HF_RE.findall(body))
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
