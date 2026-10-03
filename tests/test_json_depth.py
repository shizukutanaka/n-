"""Regression tests: deeply-nested JSON falls back instead of recursing.

json.loads raises RecursionError — a RuntimeError, not ValueError — on input
nested past the interpreter limit. Each JSONDecodeError catch that left it
out crashed the caller with a traceback instead of taking its fallback path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nmesh.bench.embed import load_embed_cache
from nmesh.bench.retrieval import load_retrieval_cache
from nmesh.eval.cache import load_eval_cache
from nmesh.eval.context import load_context_cache
from nmesh.inventory import ollama_tags
from nmesh.planner.core import load_plan
from nmesh.spec.record import load_cache as load_spec_cache
from nmesh.telemetry import Telemetry
from nmesh.watch.state import load_state


def _deep_file(tmp_path: Path) -> Path:
    depth = sys.getrecursionlimit() * 100
    target = tmp_path / "deep.json"
    target.write_text("[" * depth + "]" * depth, encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "load",
    [
        load_embed_cache,
        load_retrieval_cache,
        load_eval_cache,
        load_context_cache,
        load_spec_cache,
        load_plan,
        lambda path: Telemetry(path)._read(),
    ],
    ids=["embed", "retrieval", "eval", "context", "spec", "plan", "telemetry"],
)
def test_deeply_nested_json_falls_back(load, tmp_path: Path) -> None:
    assert not load(_deep_file(tmp_path))


def test_deeply_nested_watch_state_falls_back(tmp_path: Path) -> None:
    state = load_state(_deep_file(tmp_path))
    assert state.seen_items == {} and state.seen_findings == {}


def test_deeply_nested_ollama_manifest_skipped(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests" / "registry.ollama.ai" / "m"
    manifests.mkdir(parents=True)
    depth = sys.getrecursionlimit() * 100
    (manifests / "latest").write_text(
        "[" * depth + "]" * depth, encoding="utf-8"
    )
    assert ollama_tags(tmp_path) == {}
