"""Ground-truth verification for extracted watch mentions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import httpx

from nmesh.catalog import load_catalog
from nmesh.gateway import create_app
from nmesh.paths import nmesh_home
from nmesh.planner import BPW
from nmesh.runtime.acquisition import QUANT_ALIASES

from .extract import Mention


@dataclass(frozen=True)
class Finding:
    kind: str
    value: str
    mentions: int
    sources: tuple[str, ...]
    verified: dict[str, object]


_QUANT_RE = re.compile(r"\b(?:[I]?Q\d(?:_[A-Z0-9]+)+|fp16|bf16)\b")
_CAPS_PATH = "caps.json"


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _catalog_contains(repo: str) -> bool:
    wanted = repo.casefold()
    return any(
        wanted == source.casefold()
        for model in load_catalog()
        for source in model.sources.values()
    )


def _quant_labels(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({
        found
        for name in names
        for found in _QUANT_RE.findall(name)
    }))


def _config_fields(payload: Mapping[str, object]) -> dict[str, object]:
    source = payload.get("text_config")
    config = source if isinstance(source, Mapping) else payload
    fields: dict[str, object] = {}
    keys = (
        "architectures",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "head_dim",
        "max_position_embeddings",
    )
    for key in keys:
        value = config.get(key)
        if key == "architectures":
            if isinstance(value, list) and value and isinstance(value[0], str):
                fields[key] = value[0]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            fields[key] = value
    return fields


def _model_finding(
    mention: Mention, client: httpx.Client
) -> Finding | None:
    repo = mention.value
    try:
        response = client.get(f"https://huggingface.co/api/models/{repo}")
        if response.status_code in {401, 404}:
            return None
        response.raise_for_status()
        metadata = _mapping(response.json())
        if metadata is None:
            return None
        siblings_value = metadata.get("siblings")
        siblings = (
            tuple(
                _text(_mapping(item).get("rfilename"))
                for item in siblings_value
                if _mapping(item) is not None
            )
            if isinstance(siblings_value, list)
            else ()
        )
        gguf = tuple(name for name in siblings if name.lower().endswith(".gguf"))
        verified: dict[str, object] = {
            "downloads": metadata.get("downloads", 0),
            "likes": metadata.get("likes", 0),
            "createdAt": _text(metadata.get("createdAt")),
            "gguf_siblings": len(gguf),
            "quant_labels": list(_quant_labels(gguf)),
        }
        config_repo = repo
        config_fields: dict[str, object] = {}
        config_response = client.get(
            f"https://huggingface.co/{repo}/raw/main/config.json"
        )
        if config_response.status_code == 404 and repo.casefold().endswith("-gguf"):
            base_repo = re.sub(r"-gguf$", "", repo, flags=re.IGNORECASE)
            fallback = client.get(
                f"https://huggingface.co/{base_repo}/raw/main/config.json"
            )
            if fallback.status_code == 200:
                config_repo = base_repo
                config_response = fallback
        if config_response.status_code == 200:
            config_payload = _mapping(config_response.json())
            if config_payload is not None:
                config_fields = _config_fields(config_payload)
        elif config_response.status_code not in {401, 404}:
            config_response.raise_for_status()
        verified.update(config_fields)
        verified["config_repo"] = config_repo if config_fields else ""
        if _catalog_contains(repo):
            return None
        return Finding("catalog_gap", repo, mention.count, mention.sources, verified)
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _caps_flags() -> frozenset[str] | None:
    path = nmesh_home() / _CAPS_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict) or not entries:
            return None
        flags = {
            str(flag)
            for entry in entries.values()
            if isinstance(entry, dict)
            for flag in entry.get("flags", [])
            if isinstance(flag, str)
        }
        return frozenset(flags) if flags else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def caps_available() -> bool:
    """Return whether a recorded llama.cpp flag set can be compared."""
    return _caps_flags() is not None


def _gateway_routes() -> frozenset[str] | None:
    try:
        app = create_app()
        routes = app.routes
        paths: set[str] = set()
        for route in routes:
            try:
                path = route.path
            except AttributeError:
                continue
            if isinstance(path, str):
                paths.add(path)
        return frozenset(paths)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _known_quant(value: str) -> bool:
    lowered = value.casefold()
    return lowered in {item.casefold() for item in BPW} or lowered in {
        alias.casefold()
        for aliases in QUANT_ALIASES.values()
        for alias in aliases
    }


def verify(
    mentions: Sequence[Mention],
    client: httpx.Client,
) -> tuple[Finding, ...]:
    """Return only mentions confirmed against local or remote ground truth."""
    findings: list[Finding] = []
    flags = _caps_flags()
    routes = _gateway_routes()
    for mention in mentions:
        if mention.kind == "model_repo":
            finding = _model_finding(mention, client)
            if finding is not None:
                findings.append(finding)
        elif mention.kind == "flag" and flags is not None and mention.value not in flags:
            findings.append(Finding(
                "flag_unknown",
                mention.value,
                mention.count,
                mention.sources,
                {"backend": "llamacpp", "flag_count": len(flags)},
            ))
        elif (
            mention.kind == "route"
            and routes is not None
            and mention.value not in routes
        ):
            findings.append(Finding(
                "route_missing",
                mention.value,
                mention.count,
                mention.sources,
                {"route_count": len(routes)},
            ))
        elif mention.kind == "quant" and not _known_quant(mention.value):
            findings.append(Finding(
                "quant_unknown",
                mention.value,
                mention.count,
                mention.sources,
                {"known": sorted(BPW)},
            ))
    return tuple(findings)


__all__ = ["Finding", "caps_available", "verify"]
