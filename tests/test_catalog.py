"""Bundled catalog integrity — a hand-edited YAML must not silently degrade the planner."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from nmesh.catalog import load_catalog
from nmesh.catalog.loader import _read_models

_BUNDLED = Path(__file__).resolve().parent.parent / "nmesh" / "catalog" / "models.yaml"
_KNOWN_ROLES = {"chat", "code", "embed", "rerank", "tool"}


def test_bundled_catalog_parses_every_entry() -> None:
    raw = yaml.safe_load(_BUNDLED.read_text(encoding="utf-8"))
    parsed = _read_models(_BUNDLED)
    assert len(parsed) == len(raw)


def test_bundled_catalog_ids_are_unique() -> None:
    models = load_catalog()
    duplicates = [
        model_id
        for model_id, count in Counter(model.id for model in models).items()
        if count > 1
    ]
    assert duplicates == []


def test_bundled_catalog_invariants() -> None:
    models = load_catalog()
    assert models
    for model in models:
        assert model.id.strip() == model.id
        assert model.params > 0, model.id
        assert model.n_layers > 0, model.id
        assert model.n_heads > 0 and model.n_kv_heads > 0, model.id
        assert model.head_dim > 0 and model.hidden_size > 0, model.id
        assert model.max_context > 0, model.id
        assert model.roles, model.id
        assert set(model.roles) <= _KNOWN_ROLES, (model.id, model.roles)
        assert model.quality is None or model.quality > 0, model.id
        assert model.languages, model.id
        assert all(
            language.isalpha() and language.islower()
            for language in model.languages
        ), (model.id, model.languages)
        assert model.sources, model.id
        assert all(
            isinstance(store, str) and store
            for store in model.sources.values()
        ), model.id
        if model.sliding_window:
            assert model.sliding_window_pattern > 0, model.id
        if model.n_moe_layers:
            assert model.moe_expert_params > 0 and model.active_params > 0, model.id
