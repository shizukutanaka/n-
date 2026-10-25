from __future__ import annotations

import json
import os
import re
from pathlib import Path

from nmesh.paths import nmesh_home
from nmesh.persist import read_json_file

# Split GGUFs name every shard ``<prefix>-000NN-of-000MM.gguf``; the engine
# opens part 1 and the library reads the rest, so anything that sizes or
# verifies a model must account for all shards, not just the named file.
SPLIT_GGUF_RE = re.compile(
    r"^(?P<prefix>.+?)[-_.](?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def gguf_parts(model: Path) -> list[Path]:
    """Every numbered shard of a split GGUF, or just *model* itself."""
    match = SPLIT_GGUF_RE.match(model.name)
    if match is None:
        return [model]
    total = int(match.group("total"))
    head = model.name[: match.start("part")]
    return [
        model.parent / f"{head}{part:05d}-of-{total:05d}.gguf"
        for part in range(1, total + 1)
    ]

ArtifactCache = dict[str, int]


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
