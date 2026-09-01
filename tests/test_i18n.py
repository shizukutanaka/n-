from __future__ import annotations

import locale

from nmesh.catalog.loader import _model_from_mapping
from nmesh.i18n import MESSAGES, lang, t
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _model(model_id: str, quality: float, languages: list[str]) -> dict[str, object]:
    return {
        "id": model_id,
        "family": "test",
        "params": 1_000_000_000,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 4,
        "head_dim": 64,
        "hidden_size": 768,
        "max_context": 4096,
        "roles": ["chat"],
        "quality": quality,
        "license": "test",
        "sources": {"hf_gguf": f"{model_id}.gguf"},
        "languages": languages,
    }


def test_translation_tables_have_equal_keys() -> None:
    assert set(MESSAGES["en"]) == set(MESSAGES["ja"])


def test_translation_is_failure_tolerant() -> None:
    assert t("missing.key") == "missing.key"
    assert t("warn.language_coverage", "ja") != ""


def test_language_resolution_precedence_and_fallback(monkeypatch) -> None:
    monkeypatch.setenv("NMESH_LANG", "ja_JP.UTF-8")
    assert lang() == "ja"
    monkeypatch.setenv("NMESH_LANG", "fr")
    assert lang() == "en"
    monkeypatch.delenv("NMESH_LANG")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.setattr(locale, "getlocale", lambda: ("ja_JP", "UTF-8"))
    assert lang() == "ja"


def test_planner_warning_uses_policy_language() -> None:
    japanese = build_plan(profile(32), [], Policy(roles=["chat"], lang="ja"))
    english = build_plan(profile(32), [], Policy(roles=["chat"]))
    assert any("実行可能なモデル" in warning for warning in japanese.warnings)
    assert any("No runnable model" in warning for warning in english.warnings)


def test_language_preference_is_soft_and_normalizes_catalog_metadata() -> None:
    english = _model("english", 90, ["EN"])
    japanese = _model("japanese", 80, ["ja-JP"])
    catalog = [_model_from_mapping(english), _model_from_mapping(japanese)]
    assert catalog[0] is not None and catalog[0].languages == ("en",)
    assert catalog[1] is not None and catalog[1].languages == ("ja",)
    ordinary = build_plan(profile(32), catalog, Policy(roles=["chat"]))
    preferred = build_plan(
        profile(32), catalog, Policy(roles=["chat"], languages=("ja",))
    )
    assert ordinary.services[0].model_id == "english"
    assert preferred.services[0].model_id == "japanese"


def test_language_preference_does_not_filter_uncovered_role() -> None:
    embed = {
        **_model("english-embed", 90, ["en"]),
        "roles": ["embed"],
    }
    model = _model_from_mapping(embed)
    assert model is not None
    plan = build_plan(profile(32), [model], Policy(roles=["embed"], languages=("ja",)))
    assert plan.runnable
    assert plan.services[0].model_id == "english-embed"
    assert "english-embed" in plan.warnings[0]
    assert "publisher/vendor" in plan.warnings[0]
