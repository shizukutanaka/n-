from __future__ import annotations

import json
from pathlib import Path

import httpx

from nmesh.cli import main
from nmesh.watch.draft import write_draft
from nmesh.watch.extract import Mention, extract
from nmesh.watch.sources import SourceItem, SourceStatus, fetch_x, fetch_zenn
from nmesh.watch.state import WatchState, load_state, save_state
from nmesh.watch.verify import Finding, verify


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_zenn_fetches_article_body_not_summary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/articles":
            return httpx.Response(
                200,
                json={"articles": [{"path": "/alice/articles/watch", "title": "Watch"}]},
            )
        return httpx.Response(200, text="<main><p>--gpu-memory-utilization</p></main>")

    status, items = fetch_zenn(("llm",), 1, _client(handler))
    assert status.reachable is True
    assert status.body_available is True
    assert items[0].body.strip() == "--gpu-memory-utilization"


def test_x_without_token_reports_auth_required(monkeypatch) -> None:
    monkeypatch.delenv("NMESH_X_BEARER_TOKEN", raising=False)
    status, items = fetch_x("llm", 2)
    assert status.auth_required is True
    assert status.reachable is False
    assert status.detail == "NMESH_X_BEARER_TOKEN not set"
    assert items == ()


def test_huggingface_unauthorized_and_missing_mentions_are_dropped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    mentions = (
        Mention("model_repo", "docs/hub", 1, ("https://example/a",)),
        Mention("model_repo", "org/missing", 1, ("https://example/b",)),
    )
    with _client(handler) as client:
        assert verify(mentions, client) == ()


def test_gguf_config_fallback_records_source(monkeypatch) -> None:
    class Spec:
        def __init__(self) -> None:
            self.sources = {"hf": "other/repo"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/acme/Thing-GGUF":
            return httpx.Response(
                200,
                json={
                    "downloads": 3,
                    "likes": 2,
                    "createdAt": "2025-01-01",
                    "siblings": [{"rfilename": "thing.Q4_K_M.gguf"}],
                },
            )
        if request.url.path.endswith("/acme/Thing-GGUF/raw/main/config.json"):
            return httpx.Response(404)
        return httpx.Response(200, json={
            "architectures": ["ThingForCausalLM"],
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_size": 512,
            "max_position_embeddings": 4096,
        })

    monkeypatch.setattr("nmesh.watch.verify.load_catalog", lambda: (Spec(),))
    mentions = (Mention("model_repo", "acme/Thing-GGUF", 2, ("u",)),)
    with _client(handler) as client:
        findings = verify(mentions, client)
    assert findings[0].kind == "catalog_gap"
    assert findings[0].verified["config_repo"] == "acme/Thing"


def test_catalog_gap_is_suppressed_for_existing_source(monkeypatch) -> None:
    class Spec:
        def __init__(self) -> None:
            self.sources = {"hf": "Acme/Thing"}

    monkeypatch.setattr("nmesh.watch.verify.load_catalog", lambda: (Spec(),))
    mentions = (Mention("model_repo", "acme/thing", 1, ("u",)),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"siblings": []})

    with _client(handler) as client:
        assert verify(mentions, client) == ()


def test_huggingface_extraction_is_anchored() -> None:
    items = (SourceItem(
        "qiita",
        "https://example/item",
        "",
        "日本語 prose org/name docs/hub https://huggingface.co/acme/thing",
        "",
    ),)
    mentions = extract(items)
    assert [(item.kind, item.value) for item in mentions] == [
        ("model_repo", "acme/thing"),
    ]


def test_state_round_trip_and_dedup(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    state = WatchState("now", {"item": "now"}, {"flag|--x": "now"})
    save_state(state, path)
    loaded = load_state(path)
    assert loaded == state
    assert "flag|--x" in loaded.seen_findings
    assert load_state(tmp_path / "broken.json").seen_findings == {}


def test_draft_quality_is_null_and_missing_fields_are_explicit(tmp_path: Path) -> None:
    finding = Finding(
        "catalog_gap",
        "acme/thing",
        1,
        ("u",),
        {
            "config_repo": "acme/thing",
            "hidden_size": 512,
            "num_attention_heads": 8,
        },
    )
    path = write_draft(finding, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "quality: null" in text
    assert "missing fields:" in text
    assert "head_dim_derived: true" in text


def test_route_findings_use_app_routes(monkeypatch) -> None:
    class Route:
        def __init__(self, path: str) -> None:
            self.path = path

    class App:
        def __init__(self) -> None:
            self.routes = [Route("/v1/chat/completions")]

    monkeypatch.setattr("nmesh.watch.verify.create_app", lambda: App())
    mentions = (
        Mention("route", "/v1/models", 1, ("u",)),
        Mention("route", "/v1/chat/completions", 1, ("u",)),
    )
    with _client(lambda request: httpx.Response(500)) as client:
        findings = verify(mentions, client)
    assert [finding.value for finding in findings] == ["/v1/models"]


def test_cli_offline_json_shape_and_all_sources_unreachable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": []}), encoding="utf-8")
    monkeypatch.setenv("NMESH_HOME", str(tmp_path / "home"))
    assert main(["watch", "--offline", str(items), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "sources",
        "items",
        "mentions",
        "findings",
        "new_findings",
        "drafts",
        "notes",
    }
    monkeypatch.setattr(
        "nmesh.cli.fetch_zenn",
        lambda **kwargs: (SourceStatus("zenn", False, 0, False, False, "down"), ()),
    )
    monkeypatch.setattr(
        "nmesh.cli.fetch_qiita",
        lambda **kwargs: (SourceStatus("qiita", False, 0, False, False, "down"), ()),
    )
    assert main(["watch", "--sources", "zenn,qiita", "--json"]) == 1


def test_cli_state_deduplication_and_all_override(tmp_path: Path, monkeypatch, capsys) -> None:
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [{
        "source": "qiita",
        "url": "https://example/item",
        "body": "--gpu-memory-utilization",
    }]}), encoding="utf-8")
    monkeypatch.setenv("NMESH_HOME", str(tmp_path / "home"))
    finding = Finding(
        "flag_unknown",
        "--gpu-memory-utilization",
        1,
        ("https://example/item",),
        {"backend": "llamacpp", "flag_count": 1},
    )
    monkeypatch.setattr("nmesh.cli.verify", lambda mentions, client: (finding,))
    assert main(["watch", "--offline", str(items), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["new_findings"] == 1
    assert main(["watch", "--offline", str(items), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["new_findings"] == 0
    assert main(["watch", "--offline", str(items), "--all", "--json"]) == 0
    third = json.loads(capsys.readouterr().out)
    assert third["new_findings"] == 1
