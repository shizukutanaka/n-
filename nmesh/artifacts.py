from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from nmesh.paths import nmesh_home

ArtifactCache = dict[str, int]
CACHE_PATH = nmesh_home() / "artifacts.json"


def artifact_key(repo_id: str, label: str) -> str:
    return f"{repo_id}|{label}"


# mtime-gated parse cache, keyed by the resolved target path: acquire()
# calls load_cache() per service per pass; a stat() spots every write
# (save_cache goes through os.replace) while skipping the re-parse.
_CACHE_LOCK = threading.Lock()
_CACHED: tuple[Path, int, int, ArtifactCache] | None = None


def load_cache(path: Path | None = None) -> ArtifactCache:
    global _CACHED
    target = path or CACHE_PATH
    try:
        stamp = target.stat()
    except OSError:
        with _CACHE_LOCK:
            _CACHED = None
        return {}
    with _CACHE_LOCK:
        if (
            _CACHED is not None
            and _CACHED[0] == target
            and _CACHED[1] == stamp.st_mtime_ns
            and _CACHED[2] == stamp.st_size
        ):
            return dict(_CACHED[3])
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        values: ArtifactCache = (
            {
                str(key): int(value)
                for key, value in payload.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
            if isinstance(payload, dict)
            else {}
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        values = {}
    with _CACHE_LOCK:
        _CACHED = (target, stamp.st_mtime_ns, stamp.st_size, dict(values))
    return values


def save_cache(cache: ArtifactCache, path: Path | None = None) -> Path:
    target = path or CACHE_PATH
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
