from __future__ import annotations

import json
from pathlib import Path

from nmesh.paths import nmesh_home

ArtifactCache = dict[str, int]
CACHE_PATH = nmesh_home() / "artifacts.json"


def artifact_key(repo_id: str, label: str) -> str:
    return f"{repo_id}|{label}"


def load_cache(path: Path | None = None) -> ArtifactCache:
    target = path or CACHE_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): int(value)
            for key, value in payload.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_cache(cache: ArtifactCache, path: Path | None = None) -> Path:
    target = path or CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return target


def record(
    repo_id: str,
    label: str,
    size_bytes: int,
    path: Path | None = None,
) -> None:
    cache = load_cache(path)
    cache[artifact_key(repo_id, label)] = int(size_bytes)
    save_cache(cache, path)
