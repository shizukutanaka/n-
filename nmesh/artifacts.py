from __future__ import annotations

import json
import os
from pathlib import Path

from nmesh.paths import nmesh_home
from nmesh.persist import read_json_file

ArtifactCache = dict[str, int]
# Retained for API compatibility; functions resolve nmesh_home() lazily so a
# NMESH_HOME set after import is honored.
CACHE_PATH = nmesh_home() / "artifacts.json"


def artifact_key(repo_id: str, label: str) -> str:
    return f"{repo_id}|{label}"


def load_cache(path: Path | None = None) -> ArtifactCache:
    target = path or (nmesh_home() / "artifacts.json")
    payload = read_json_file(target)
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in payload.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def save_cache(cache: ArtifactCache, path: Path | None = None) -> Path:
    target = path or (nmesh_home() / "artifacts.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    os.replace(temporary, target)
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
