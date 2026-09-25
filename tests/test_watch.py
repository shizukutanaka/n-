from __future__ import annotations

import json
from pathlib import Path

import httpx

from nmesh.cli import _candidate_fit, main
from nmesh.watch.draft import write_draft
from nmesh.watch.extract import Mention, extract
from nmesh.watch.sources import (
    _GITHUB_REPOS,
    _QIITA_TAGS,
    _ZENN_TOPICS,
    SourceItem,
    SourceStatus,
    fetch_arxiv,
    fetch_github,
    fetch_hf,
    fetch_qiita,
    fetch_x,
    fetch_zenn,
)
from nmesh.watch.state import WatchState, load_state, save_state
from nmesh.watch.verify import Finding, _known_quant, _weight_sets, verify


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_known_quant_requires_planner_supported_canonical_label() -> None:
    assert _known_quant("Q4_K_M") is True
    assert _known_quant("q4_k_m") is True
    assert _known_quant("fp16") is True
    assert _known_quant("IQ4_XS") is False
    assert _known_quant("Q4_K_XL") is False
    assert _known_quant("Q3_K_L") is False
    assert _known_quant("q3_k_l+q8") is False


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


def test_source_topic_constants_use_verified_names() -> None:
    assert _ZENN_TOPICS == ("llm", "ollama", "llamacpp", "vllm", "gguf", "localllm")
    assert "llama.cpp" in _QIITA_TAGS
    assert "llamacpp" in _QIITA_TAGS


def test_zenn_limit_is_per_topic_and_zero_yields_are_visible() -> None:
    topics: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        topic = request.url.params["topicname"]
        topics.append(topic)
        if topic == "empty":
            return httpx.Response(200, json={"articles": []})
        return httpx.Response(200, json={
            "articles": [
                {"path": f"/a/{topic}/one", "title": topic},
                {"path": f"/a/{topic}/two", "title": topic},
            ],
        })

    def combined(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/articles":
            return handler(request)
        return httpx.Response(200, text="<p>body</p>")

    status, items = fetch_zenn(("llm", "empty", "gguf"), 1, _client(combined))
    assert topics == ["llm", "empty", "gguf"]
    assert len(items) == 2
    assert status.detail == "llm=1; empty=0; gguf=1"


def test_qiita_limit_is_per_tag() -> None:
    tags: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        tag = query.removeprefix("tag:")
        tags.append(tag)
        assert request.url.params["per_page"] == "1"
        return httpx.Response(200, json=[
            {"url": f"https://qiita.com/{tag}/one", "title": tag, "body": ""},
        ])

    status, items = fetch_qiita(("llama.cpp", "llamacpp", "localllm"), 1, _client(handler))
    assert tags == ["llama.cpp", "llamacpp", "localllm"]
    assert status.items == 3
    assert len(items) == 3


def test_github_fetches_release_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.github.com"
        return httpx.Response(200, json=[{
            "html_url": "https://github.com/o/r/releases/tag/b1",
            "name": "b1",
            "tag_name": "b1",
            "body": "server : add --cpu-moe flag for MoE offload",
            "published_at": "2026-01-01T00:00:00Z",
            "draft": False,
        }])
    status, items = fetch_github(("o/r",), 5, _client(handler))
    assert status.reachable and status.items == 1 and status.body_available
    assert items[0].source == "github"
    assert "--cpu-moe" in items[0].body


def test_github_rate_limit_reports_unreachable_with_hint(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"message": "API rate limit exceeded"},
            headers={"x-ratelimit-remaining": "0"},
        )
    status, items = fetch_github(("o/r",), 5, _client(handler))
    assert not status.reachable and items == ()
    assert "GITHUB_TOKEN" in status.detail


def test_arxiv_parses_atom_entries() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2601.00001v1</id>
        <title>  Efficient\nKV Cache Compression </title>
        <summary> We study speculative\n decoding for LLM inference. </summary>
        <published>2026-01-02T00:00:00Z</published>
        <link href="http://arxiv.org/abs/2601.00001v1"/>
      </entry>
    </feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(200, text=feed)
    status, items = fetch_arxiv('cat:cs.CL AND all:"kv cache"', 5, _client(handler))
    assert status.reachable and len(items) == 1
    assert items[0].source == "arxiv"
    assert items[0].title == "Efficient KV Cache Compression"
    assert "speculative decoding" in items[0].body


def test_arxiv_throttled_reports_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)
    status, items = fetch_arxiv("all:x", 1, _client(handler))
    assert not status.reachable and items == ()


def test_github_source_constant_is_verified_engine_repos() -> None:
    assert "ggml-org/llama.cpp" in _GITHUB_REPOS


def test_hf_fetches_newest_gguf_models_and_cards() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models":
            assert request.url.params["filter"] == "gguf"
            assert request.url.params["sort"] == "lastModified"
            assert request.url.params["direction"] == "-1"
            return httpx.Response(200, json=[
                {"id": "acme/NewModel-GGUF", "lastModified": "2026-01-02T00:00:00Z"},
                {"id": "acme/NoCard-GGUF", "lastModified": "2026-01-01T00:00:00Z"},
            ])
        if request.url.path == "/acme/NewModel-GGUF/raw/main/README.md":
            return httpx.Response(200, text="quantized to Q4_K_M by acme")
        return httpx.Response(404)

    status, items = fetch_hf("gguf", 5, _client(handler))
    assert status.reachable and status.items == 2 and status.body_available
    assert status.detail == "2 models; 1 cards"
    assert items[0].source == "hf"
    assert items[0].url == "https://huggingface.co/acme/NewModel-GGUF"
    assert "Q4_K_M" in items[0].body
    assert items[1].body == ""


def test_hf_unreachable_is_reported_honestly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    status, items = fetch_hf("gguf", 5, _client(handler))
    assert not status.reachable and items == ()


def test_zenn_article_page_failure_skips_only_that_article() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/articles":
            return httpx.Response(200, json={"articles": [
                {"path": "/a/gone", "title": "gone"},
                {"path": "/a/ok", "title": "ok"},
            ]})
        if request.url.path == "/a/gone":
            return httpx.Response(404)
        return httpx.Response(200, text="<p>kv cache</p>")

    status, items = fetch_zenn(("llm",), 2, _client(handler))
    assert status.reachable and len(items) == 1
    assert items[0].url == "https://zenn.dev/a/ok"


def test_hf_card_fetch_error_keeps_the_model_listing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models":
            return httpx.Response(200, json=[
                {"id": "acme/Model-GGUF", "lastModified": "2026-01-02T00:00:00Z"},
            ])
        raise httpx.ConnectError("connection lost", request=request)

    status, items = fetch_hf("gguf", 5, _client(handler))
    assert status.reachable and len(items) == 1
    assert items[0].body == ""


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


def test_huggingface_metadata_and_weight_sets_are_verified(monkeypatch) -> None:
    monkeypatch.setattr("nmesh.watch.verify.load_catalog", lambda: ())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/acme/model":
            return httpx.Response(200, json={
                "pipeline_tag": "text-generation",
                "gated": False,
                "cardData": {"license": "apache-2.0"},
                "safetensors": {"total": 123},
                "siblings": [{"rfilename": "model.Q4_K_M.gguf"}],
            })
        if "/tree/main" in str(request.url):
            return httpx.Response(200, json=[
                {"path": "model.Q4_K_M-00001-of-00002.gguf", "size": 4},
                {"path": "model.Q4_K_M-00002-of-00002.gguf", "lfs": {"size": 6}},
            ])
        if request.url.path.endswith("/raw/main/config.json"):
            return httpx.Response(200, json={
                "architectures": ["ModelForCausalLM"],
                "num_hidden_layers": 2,
            })
        raise AssertionError(request.url)

    mention = Mention("model_repo", "acme/model", 1, ("u",))
    with _client(handler) as client:
        finding = verify((mention,), client)[0]
    assert finding.verified["pipeline_tag"] == "text-generation"
    assert finding.verified["gated"] is False
    assert finding.verified["license"] == "apache-2.0"
    assert finding.verified["params"] == 123
    assert finding.verified["weight_sets"] == {".:Q4_K_M": 10}
    assert finding.verified["smallest_weight_bytes"] == 10


def test_catalog_gap_is_suppressed_for_existing_source(monkeypatch) -> None:
    class Spec:
        def __init__(self) -> None:
            self.sources = {"hf": "Acme/Thing"}

    monkeypatch.setattr("nmesh.watch.verify.load_catalog", lambda: (Spec(),))
    mentions = (Mention("model_repo", "acme/thing", 1, ("u",)),)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("catalog-known repository should not query Hugging Face")

    with _client(handler) as client:
        assert verify(mentions, client) == ()


def test_catalog_metrics_partition_mentioned_repositories(monkeypatch) -> None:
    class Spec:
        def __init__(self) -> None:
            self.sources = {"hf": "known/repo"}

    monkeypatch.setattr("nmesh.watch.verify.load_catalog", lambda: (Spec(),))
    mentions = (
        Mention("model_repo", "known/repo", 1, ("known",)),
        Mention("model_repo", "absent/repo", 1, ("absent",)),
        Mention("model_repo", "unresolved/repo", 1, ("unresolved",)),
    )
    stats = {"mentioned_repo_ids": 3, "in_catalog": 0, "resolved_repo_ids": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/models/absent/repo"):
            return httpx.Response(200, json={"siblings": []})
        if request.url.path.endswith("/api/models/unresolved/repo"):
            return httpx.Response(404)
        if request.url.path.endswith("/raw/main/config.json"):
            return httpx.Response(404)
        if "/tree/main" in str(request.url):
            return httpx.Response(200, json=[])
        raise AssertionError("catalog-known repository should not query Hugging Face")

    with _client(handler) as client:
        findings = verify(mentions, client, stats)
    absent = len([finding for finding in findings if finding.kind == "catalog_gap"])
    unresolved = stats["mentioned_repo_ids"] - stats["in_catalog"] - absent
    assert stats == {
        "mentioned_repo_ids": 3,
        "in_catalog": 1,
        "resolved_repo_ids": 1,
    }
    assert stats["in_catalog"] + absent + unresolved == stats["mentioned_repo_ids"]


def test_huggingface_extraction_is_anchored() -> None:
    items = (SourceItem(
        "qiita",
        "https://example/item",
        "",
        "日本語 prose org/name https://huggingface.co/docs/hub "
        "https://huggingface.co/papers/2504.13181 "
        "https://huggingface.co/datasets/leemeng "
        "https://huggingface.co/blog/nvidia "
        "https://huggingface.co/acme/thing",
        "",
    ),)
    mentions = extract(items)
    assert [item.value for item in mentions] == [
        "acme/thing",
        "blog/nvidia",
        "datasets/leemeng",
        "docs/hub",
        "papers/2504.13181",
    ]
    with _client(lambda request: httpx.Response(404)) as client:
        assert verify(mentions, client) == ()


def test_flag_extraction_is_limited_to_llamacpp_commands() -> None:
    """caps.json holds llama.cpp flags, so other tools' flags are not comparable."""
    items = (SourceItem(
        "qiita",
        "https://example/item",
        "",
        "git diff --no-ext-diff\n"
        "npm ci --frozen-lockfile\n"
        "./llama-server -m model.gguf --jinja --cache-type-k q8_0\n"
        "docker run --gpus all vllm/vllm-openai --max-model-len 8192\n"
        "systemctl disable llama-server --now\n"
        "llama-bench \\\n  --n-gpu-layers 99\n",
        "",
    ),)
    assert [
        mention.value for mention in extract(items) if mention.kind == "flag"
    ] == ["--cache-type-k", "--jinja", "--n-gpu-layers"]


def test_flag_finding_records_caps_binaries(tmp_path: Path, monkeypatch) -> None:
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps({
        "entries": {
            "C:/llama-a.exe": {"flags": ["--known"]},
            "C:/llama-b.exe": {"flags": ["--other", "--known"]},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    monkeypatch.setattr("nmesh.watch.verify._gateway_routes", lambda: frozenset())
    mentions = (Mention("flag", "--unknown", 1, ("u",)),)
    with _client(lambda request: httpx.Response(500)) as client:
        findings = verify(mentions, client)
    assert findings[0].verified == {
        "binaries": [
            {"path": "C:/llama-a.exe", "flag_count": 1},
            {"path": "C:/llama-b.exe", "flag_count": 2},
        ],
        "flag_count": 2,
    }


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


def test_weight_sets_sum_shards_and_exclude_auxiliaries() -> None:
    payload = [
        {"path": "Q4_K_M/model.Q4_K_M-00001-of-00002.gguf", "size": 10},
        {"path": "Q4_K_M/model.Q4_K_M-00002-of-00002.gguf", "lfs": {"size": 12}},
        {"path": "Q4_K_M/model-mmproj-f16.gguf", "size": 1},
        {"path": "Q4_K_M/model-imatrix.gguf", "size": 1},
        {"path": "Q4_K_M/model-MTP.gguf", "size": 1},
        {"path": "Q4_K_M/model-draft.gguf", "size": 1},
        {"path": "Q4_K_M/model-vocab.gguf", "size": 1},
    ]
    assert _weight_sets(payload) == {"Q4_K_M:Q4_K_M": 22}
    assert _weight_sets([
        {"path": "model-mmproj-f16.gguf", "size": 1},
        {"path": "model-vocab.gguf", "size": 1},
    ]) == {}


def test_draft_includes_verified_metadata_without_quality(tmp_path: Path) -> None:
    finding = Finding(
        "catalog_gap",
        "acme/thing",
        1,
        ("u",),
        {
            "architectures": "ThingForCausalLM",
            "params": 123,
            "license": "apache-2.0",
            "pipeline_tag": "text-generation",
            "gated": False,
            "smallest_weight_bytes": 456,
            "config_repo": "acme/thing",
        },
    )
    text = write_draft(finding, tmp_path).read_text(encoding="utf-8")
    assert "params: 123" in text
    assert 'license: "apache-2.0"' in text
    assert 'roles: ["chat"]' in text
    assert "quality: null" in text
    assert "# pipeline_tag: \"text-generation\"" in text
    assert "# gated: false" in text
    assert "# smallest_weight_bytes: 456" in text


def test_candidate_fit_order_is_explicit() -> None:
    def finding(verified: dict[str, object]) -> Finding:
        return Finding("catalog_gap", "org/model", 1, ("u",), verified)

    assert _candidate_fit(finding({"weight_sets": {}}), 10) == "no_weights"
    assert _candidate_fit(
        finding({"weight_sets": {"q": 1}, "gated": True}), 10
    ) == "gated"
    assert _candidate_fit(finding({"weight_sets": {"q": 1}}), 10) == "role_unknown"
    assert _candidate_fit(
        finding({"weight_sets": {"q": 1}, "pipeline_tag": "image-to-text"}), 10
    ) == "not_text"
    assert _candidate_fit(
        finding({
            "weight_sets": {"q": 20},
            "pipeline_tag": "text-generation",
            "smallest_weight_bytes": 20,
        }),
        10,
    ) == "too_large"
    assert _candidate_fit(
        finding({
            "weight_sets": {"q": 1},
            "pipeline_tag": "text-generation",
            "smallest_weight_bytes": 1,
        }),
        10,
    ) == "fits"


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
        "catalog",
        "candidates",
    }
    assert set(payload["catalog"]) == {
        "entries",
        "repo_ids",
        "mentioned_repo_ids",
        "in_catalog",
        "resolved_repo_ids",
        "absent_repo_ids",
    }
    assert set(payload["candidates"]) == {
        "total", "counts", "fits", "budget_bytes", "budget_source",
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
    monkeypatch.setattr(
        "nmesh.cli.verify",
        lambda mentions, client, catalog_stats: (finding,),
    )
    assert main(["watch", "--offline", str(items), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["new_findings"] == 1
    assert main(["watch", "--offline", str(items), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["new_findings"] == 0
    assert main(["watch", "--offline", str(items), "--all", "--json"]) == 0
    third = json.loads(capsys.readouterr().out)
    assert third["new_findings"] == 1
