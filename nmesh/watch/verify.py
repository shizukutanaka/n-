"""Ground-truth verification for extracted watch mentions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

import httpx

from nmesh.catalog import load_catalog
from nmesh.gateway import create_app
from nmesh.paths import nmesh_home
from nmesh.planner import BPW
from nmesh.runtime.acquisition import parse_label

from .extract import Mention


@dataclass(frozen=True)
class Finding:
    kind: str
    value: str
    mentions: int
    sources: tuple[str, ...]
    verified: dict[str, object]


_QUANT_RE = re.compile(r"\b(?:[I]?Q\d(?:_[A-Z0-9]+)+|fp16|bf16)\b")
_WEIGHT_QUANT_RE = re.compile(
    r"(?:UD-Q[A-Z0-9._-]+|IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|fp16)",
    re.IGNORECASE,
)
_AUXILIARY_RE = re.compile(r"mmproj|imatrix|mtp|draft|vocab", re.IGNORECASE)
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


def _metadata_fields(metadata: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key in ("pipeline_tag", "gated"):
        if key in metadata and isinstance(metadata[key], (str, bool)):
            fields[key] = metadata[key]
    card_data = _mapping(metadata.get("cardData"))
    license_value = card_data.get("license") if card_data is not None else None
    if not isinstance(license_value, str):
        tags = metadata.get("tags")
        license_value = next(
            (
                tag.removeprefix("license:")
                for tag in tags
                if isinstance(tag, str) and tag.casefold().startswith("license:")
            ),
            None,
        ) if isinstance(tags, list) else None
    if isinstance(license_value, str) and license_value:
        fields["license"] = license_value
    safetensors = _mapping(metadata.get("safetensors"))
    params = safetensors.get("total") if safetensors is not None else None
    if isinstance(params, (int, float)) and not isinstance(params, bool):
        fields["params"] = params
    return fields


def _weight_sets(payload: object) -> dict[str, int]:
    if not isinstance(payload, list):
        return {}
    totals: dict[str, int] = {}
    for item in payload:
        entry = _mapping(item)
        if entry is None:
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.casefold().endswith(".gguf"):
            continue
        if _AUXILIARY_RE.search(PurePosixPath(path).name):
            continue
        quant = _WEIGHT_QUANT_RE.search(PurePosixPath(path).stem)
        if quant is None:
            continue
        lfs = _mapping(entry.get("lfs"))
        size = lfs.get("size") if lfs is not None else entry.get("size")
        if not isinstance(size, (int, float)) or isinstance(size, bool):
            continue
        stem = re.sub(
            r"-\d{5}-of-\d{5}\.gguf$",
            ".gguf",
            path,
            flags=re.IGNORECASE,
        )
        parent = PurePosixPath(stem).parent.as_posix()
        key = f"{parent}:{quant.group(0).upper()}"
        totals[key] = totals.get(key, 0) + int(size)
    return totals


def _tree_weight_sets(repo: str, client: httpx.Client) -> dict[str, int]:
    try:
        response = client.get(
            f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
        )
        if response.status_code != 200:
            return {}
        return _weight_sets(response.json())
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _model_finding(
    mention: Mention,
    client: httpx.Client,
    stats: dict[str, int] | None = None,
) -> Finding | None:
    repo = mention.value
    try:
        if _catalog_contains(repo):
            if stats is not None:
                stats["in_catalog"] = stats.get("in_catalog", 0) + 1
            return None
        response = client.get(f"https://huggingface.co/api/models/{repo}")
        if response.status_code in {401, 404}:
            return None
        response.raise_for_status()
        metadata = _mapping(response.json())
        if metadata is None:
            return None
        if stats is not None:
            stats["resolved_repo_ids"] = stats.get("resolved_repo_ids", 0) + 1
        siblings_value = metadata.get("siblings")
        siblings = (
            tuple(
                _text(sibling.get("rfilename"))
                for item in siblings_value
                if (sibling := _mapping(item)) is not None
            )
            if isinstance(siblings_value, list)
            else ()
        )
        gguf = tuple(name for name in siblings if name.lower().endswith(".gguf"))
        verified: dict[str, object] = _metadata_fields(metadata)
        verified.update({
            "downloads": metadata.get("downloads", 0),
            "likes": metadata.get("likes", 0),
            "createdAt": _text(metadata.get("createdAt")),
            "gguf_siblings": len(gguf),
            "quant_labels": list(_quant_labels(gguf)),
        })
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
                base_metadata_response = client.get(
                    f"https://huggingface.co/api/models/{base_repo}"
                )
                if base_metadata_response.status_code == 200:
                    base_metadata = _mapping(base_metadata_response.json())
                    if base_metadata is not None:
                        for key, value in _metadata_fields(base_metadata).items():
                            verified.setdefault(key, value)
        if config_response.status_code == 200:
            config_payload = _mapping(config_response.json())
            if config_payload is not None:
                config_fields = _config_fields(config_payload)
        elif config_response.status_code not in {401, 404}:
            config_response.raise_for_status()
        verified.update(config_fields)
        verified["config_repo"] = config_repo if config_fields else ""
        weight_sets = _tree_weight_sets(repo, client)
        verified["weight_sets"] = weight_sets
        if weight_sets:
            verified["smallest_weight_bytes"] = min(weight_sets.values())
        return Finding("catalog_gap", repo, mention.count, mention.sources, verified)
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _caps_flags() -> tuple[frozenset[str], tuple[dict[str, object], ...]] | None:
    path = nmesh_home() / _CAPS_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict) or not entries:
            return None
        combined: set[str] = set()
        binaries: list[dict[str, object]] = []
        for entry_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            raw_flags = entry.get("flags")
            if not isinstance(raw_flags, list):
                continue
            flags = {flag for flag in raw_flags if isinstance(flag, str)}
            if not flags:
                continue
            combined.update(flags)
            binary = entry.get("binary")
            binaries.append({
                "path": binary if isinstance(binary, str) else str(entry_key),
                "flag_count": len(flags),
            })
        if not combined:
            return None
        return frozenset(combined), tuple(binaries)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def caps_available() -> bool:
    """Return whether a recorded llama.cpp flag set can be compared."""
    return _caps_flags() is not None


@runtime_checkable
class _RoutePath(Protocol):
    path: str


def _gateway_routes() -> frozenset[str] | None:
    try:
        app = create_app()
        return frozenset(
            route.path
            for route in app.routes
            if isinstance(route, _RoutePath) and isinstance(route.path, str)
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _known_quant(value: str) -> bool:
    canonical = parse_label(f"{value}.gguf")
    return canonical in BPW if canonical is not None else False


def verify(
    mentions: Sequence[Mention],
    client: httpx.Client,
    catalog_stats: dict[str, int] | None = None,
) -> tuple[Finding, ...]:
    """Return only mentions confirmed against local or remote ground truth."""
    findings: list[Finding] = []
    flags_data = _caps_flags()
    flags = flags_data[0] if flags_data is not None else None
    routes = _gateway_routes()
    for mention in mentions:
        if mention.kind == "model_repo":
            finding = _model_finding(mention, client, catalog_stats)
            if finding is not None:
                findings.append(finding)
        elif mention.kind == "flag" and flags is not None and mention.value not in flags:
            findings.append(Finding(
                "flag_unknown",
                mention.value,
                mention.count,
                mention.sources,
                {
                    "binaries": list(flags_data[1]) if flags_data is not None else [],
                    "flag_count": len(flags),
                },
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
