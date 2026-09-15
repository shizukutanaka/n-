"""Network source adapters for the evidence-gated watch layer.

Qiita exposes full markdown bodies. Zenn's RSS is summary-only, so the
unofficial article API is followed by HTML fetches. X is intentionally
unavailable without an explicit bearer token.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SourceItem:
    source: str
    url: str
    title: str
    body: str
    published: str


@dataclass(frozen=True)
class SourceStatus:
    name: str
    reachable: bool
    items: int
    body_available: bool
    auth_required: bool
    detail: str


_TAG_RE = re.compile(r"<[^>]+>")
_ZENN_TOPICS = ("llm", "ollama", "llamacpp", "vllm", "gguf", "localllm")
_QIITA_TAGS = (
    "llm",
    "ollama",
    "llama.cpp",
    "llamacpp",
    "vllm",
    "gguf",
    "localllm",
)


def _failure(name: str, error: object, auth_required: bool = False) -> tuple[
    SourceStatus, tuple[SourceItem, ...]
]:
    return SourceStatus(name, False, 0, False, auth_required, str(error)), ()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _unique_items(items: Sequence[SourceItem]) -> tuple[SourceItem, ...]:
    unique: dict[str, SourceItem] = {}
    for item in items:
        unique.setdefault(item.url, item)
    return tuple(unique.values())


def fetch_qiita(
    tags: Sequence[str] = _QIITA_TAGS,
    limit: int = 20,
    client: httpx.Client | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch full Qiita item bodies for the requested tags."""
    own_client = client is None
    session = client or httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        headers = {}
        token = os.environ.get("QIITA_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        items: list[SourceItem] = []
        for tag in tags:
            response = session.get(
                "https://qiita.com/api/v2/items",
                params={"per_page": limit, "query": f"tag:{tag}"},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("Qiita response was not a list")
            for raw in payload:
                item = _mapping(raw)
                if item is None:
                    continue
                url = _text(item.get("url"))
                if not url:
                    continue
                items.append(SourceItem(
                    "qiita",
                    url,
                    _text(item.get("title")),
                    _text(item.get("body")),
                    _text(item.get("created_at")),
                ))
        selected = _unique_items(items)
        return SourceStatus("qiita", True, len(selected), True, False, ""), selected
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return _failure("qiita", error)
    finally:
        if own_client:
            session.close()


def _strip_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).replace("\xa0", " ")


def fetch_zenn(
    topics: Sequence[str] = _ZENN_TOPICS,
    limit: int = 20,
    client: httpx.Client | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch Zenn article pages because the RSS summaries lack technical text."""
    own_client = client is None
    session = client or httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        items: list[SourceItem] = []
        yields: list[str] = []
        for topic in topics:
            response = session.get(
                "https://zenn.dev/api/articles",
                params={
                    "topicname": topic,
                    "order": "latest",
                },
            )
            response.raise_for_status()
            payload = _mapping(response.json())
            raw_articles = payload.get("articles") if payload is not None else None
            if not isinstance(raw_articles, list):
                raise TypeError("Zenn response did not contain articles")
            articles = raw_articles[:limit]
            yields.append(f"{topic}={len(articles)}")
            for raw in articles:
                article = _mapping(raw)
                if article is None:
                    continue
                path = _text(article.get("path"))
                if not path:
                    continue
                url = f"https://zenn.dev{path}"
                page = session.get(url)
                page.raise_for_status()
                items.append(SourceItem(
                    "zenn",
                    url,
                    _text(article.get("title")),
                    _strip_html(page.text),
                    _text(article.get("published_at") or article.get("publishedAt")),
                ))
        selected = _unique_items(items)
        return SourceStatus(
            "zenn",
            True,
            len(selected),
            True,
            False,
            "; ".join(yields),
        ), selected
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return _failure("zenn", error)
    finally:
        if own_client:
            session.close()


def fetch_x(
    query: str,
    limit: int = 20,
    client: httpx.Client | None = None,
    bearer: str | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch recent X posts, honestly reporting that authentication is required."""
    token = bearer or os.environ.get("NMESH_X_BEARER_TOKEN")
    if not token:
        return _failure("x", "NMESH_X_BEARER_TOKEN not set", True)
    own_client = client is None
    session = client or httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        response = session.get(
            "https://api.x.com/2/tweets/search/recent",
            params={"query": query, "max_results": min(max(limit, 10), 100)},
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code == 401:
            return _failure("x", "HTTP 401 Unauthorized", True)
        response.raise_for_status()
        payload = _mapping(response.json())
        rows = payload.get("data") if payload is not None else None
        if not isinstance(rows, list):
            rows = []
        items = []
        for raw in rows[:limit]:
            tweet = _mapping(raw)
            if tweet is None:
                continue
            tweet_id = _text(tweet.get("id"))
            if not tweet_id:
                continue
            items.append(SourceItem(
                "x",
                f"https://x.com/i/web/status/{tweet_id}",
                "",
                _text(tweet.get("text")),
                _text(tweet.get("created_at")),
            ))
        selected = tuple(items)
        return SourceStatus("x", True, len(selected), True, False, ""), selected
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return _failure("x", error)
    finally:
        if own_client:
            session.close()


__all__ = [
    "SourceItem",
    "SourceStatus",
    "fetch_qiita",
    "fetch_x",
    "fetch_zenn",
]
