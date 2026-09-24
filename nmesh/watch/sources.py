"""Network source adapters for the evidence-gated watch layer.

Qiita exposes full markdown bodies. Zenn's RSS is summary-only, so the
unofficial article API is followed by HTML fetches. X is intentionally
unavailable without an explicit bearer token.
"""

from __future__ import annotations

import html
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlencode

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
_GITHUB_REPOS = ("ggml-org/llama.cpp", "vllm-project/vllm", "ollama/ollama")
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_QUERY = (
    'cat:cs.CL AND (all:"kv cache" OR all:"speculative decoding" '
    'OR all:"llm inference" OR all:gguf OR all:"llama.cpp")'
)
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
        def fetch_tag(tag: str) -> list[SourceItem]:
            response = session.get(
                "https://qiita.com/api/v2/items",
                params={"per_page": limit, "query": f"tag:{tag}"},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("Qiita response was not a list")
            out: list[SourceItem] = []
            for raw in payload:
                item = _mapping(raw)
                if item is None:
                    continue
                url = _text(item.get("url"))
                if not url:
                    continue
                out.append(SourceItem(
                    "qiita",
                    url,
                    _text(item.get("title")),
                    _text(item.get("body")),
                    _text(item.get("created_at")),
                ))
            return out

        # Each tag listing is an independent read — fetch concurrently
        # (httpx.Client is thread-safe), merging in the requested order.
        workers = min(6, len(tags))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                batches = list(pool.map(fetch_tag, tags))
        else:
            batches = [fetch_tag(tag) for tag in tags]
        items: list[SourceItem] = [item for batch in batches for item in batch]
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


def fetch_github(
    repos: Sequence[str] = _GITHUB_REPOS,
    limit: int = 10,
    client: httpx.Client | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch release notes for the watched engines via the GitHub REST API.

    The unauthenticated API is rate-limited per IP (60/h); GITHUB_TOKEN or
    GH_TOKEN raises that ceiling and also fixes shared-egress boxes where
    the anonymous quota is already spent.
    """
    own_client = client is None
    session = client or httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        def fetch_repo(
            repo: str,
        ) -> tuple[list[SourceItem] | None, str, str]:
            """Fetch one repo's releases; items None marks a bare-IP 403."""
            response = session.get(
                f"https://api.github.com/repos/{repo}/releases",
                params={"per_page": min(max(limit, 1), 100)},
                headers=headers,
            )
            if response.status_code == 403 and not token:
                return None, "", response.headers.get(
                    "x-ratelimit-remaining", ""
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("GitHub response was not a list")
            out: list[SourceItem] = []
            for raw in payload:
                release = _mapping(raw)
                if release is None or release.get("draft"):
                    continue
                url = _text(release.get("html_url"))
                if not url:
                    continue
                out.append(SourceItem(
                    "github",
                    url,
                    _text(release.get("name") or release.get("tag_name")),
                    _text(release.get("body")),
                    _text(release.get("published_at")),
                ))
            return out, f"{repo}={len(out)}", ""

        # Each repo's release listing is an independent read — fetch
        # concurrently (httpx.Client is thread-safe), merging in order.
        workers = min(6, len(repos))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fetched_list = list(pool.map(fetch_repo, repos))
        else:
            fetched_list = [fetch_repo(repo) for repo in repos]
        # The serial loop bailed on the first unauthenticated 403 — keep that
        # precedence (first repo order) after the pool joins.
        rate_limited = next(
            (remaining for batch, _, remaining in fetched_list
             if batch is None),
            None,
        )
        if rate_limited is not None:
            return _failure(
                "github",
                f"GitHub API 403 {rate_limited}; "
                "set GITHUB_TOKEN to authenticate",
            )
        items = [
            item
            for batch, _, _ in fetched_list
            for item in (batch or ())
        ]
        yields = [yield_text for _, yield_text, _ in fetched_list]
        selected = _unique_items(items)
        return SourceStatus(
            "github",
            True,
            len(selected),
            any(item.body for item in selected),
            False,
            "; ".join(yields),
        ), selected
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return _failure("github", error)
    finally:
        if own_client:
            session.close()


def fetch_arxiv(
    query: str = _ARXIV_QUERY,
    limit: int = 20,
    client: httpx.Client | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch recent arXiv cs.CL abstracts on local-inference topics.

    arXiv's Atom API expects `+`-joined search terms and throttles bursts;
    failures are reported as unreachable rather than retried.
    """
    own_client = client is None
    session = client or httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        # arXiv wants '+'-joined search terms with ':' left readable, so the
        # request URL is encoded by hand rather than through httpx params.
        params = urlencode(
            {
                "search_query": query,
                "start": 0,
                "max_results": min(max(limit, 1), 50),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            safe=":",
        )
        response = session.get(f"https://export.arxiv.org/api/query?{params}")
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items: list[SourceItem] = []
        for entry in root.findall("atom:entry", _ARXIV_NS):
            url = ""
            for link in entry.findall("atom:link", _ARXIV_NS):
                if link.get("href"):
                    url = link.get("href", "")
                    break
            if not url:
                url = _text(entry.findtext("atom:id", default="", namespaces=_ARXIV_NS))
            if not url:
                continue
            title = _text(entry.findtext("atom:title", default="", namespaces=_ARXIV_NS))
            summary = _text(entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS))
            published = _text(entry.findtext("atom:published", default="", namespaces=_ARXIV_NS))
            items.append(SourceItem(
                "arxiv",
                url,
                " ".join(title.split()),
                " ".join(summary.split()),
                published,
            ))
        return SourceStatus(
            "arxiv", True, len(items), bool(items), False, ""
        ), tuple(items)
    except (httpx.HTTPError, ValueError, TypeError, ET.ParseError) as error:
        return _failure("arxiv", error)
    finally:
        if own_client:
            session.close()


_HF_TAG = "gguf"
# Model-card reads are capped: cards can exceed a megabyte and only their
# head carries the metadata extraction needs (quants, repos, commands).
_HF_CARD_BYTES = 49152


def fetch_hf(
    tag: str = _HF_TAG,
    limit: int = 8,
    client: httpx.Client | None = None,
) -> tuple[SourceStatus, tuple[SourceItem, ...]]:
    """Fetch the newest GGUF-tagged Hugging Face models plus their cards.

    The Hub API allows anonymous reads (a shared quota); HF_TOKEN or
    HUGGING_FACE_HUB_TOKEN raises the rate ceiling. A model without a
    README.md still lists with an empty body rather than being skipped.
    """
    own_client = client is None
    session = client or httpx.Client(timeout=15.0, follow_redirects=True)
    try:
        headers = {"Accept": "application/json"}
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = session.get(
            "https://huggingface.co/api/models",
            params={
                "filter": tag,
                "sort": "lastModified",
                "direction": "-1",
                "limit": min(max(limit, 1), 50),
            },
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise TypeError("Hugging Face response was not a list")
        items: list[SourceItem] = []
        cards = 0
        for raw in payload:
            entry = _mapping(raw)
            if entry is None:
                continue
            model_id = _text(entry.get("id") or entry.get("modelId"))
            if not model_id:
                continue
            url = f"https://huggingface.co/{model_id}"
            card = session.get(f"{url}/raw/main/README.md", headers=headers)
            body = card.text[:_HF_CARD_BYTES] if card.status_code == 200 else ""
            if body:
                cards += 1
            items.append(SourceItem(
                "hf",
                url,
                model_id,
                body,
                _text(entry.get("lastModified") or entry.get("last_modified")),
            ))
        selected = _unique_items(items)
        return SourceStatus(
            "hf",
            True,
            len(selected),
            cards > 0,
            False,
            f"{len(selected)} models; {cards} cards",
        ), selected
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return _failure("hf", error)
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
    "fetch_arxiv",
    "fetch_github",
    "fetch_hf",
    "fetch_qiita",
    "fetch_x",
    "fetch_zenn",
]
