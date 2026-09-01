from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

CAPS_PATH = Path.home() / ".nmesh" / "caps.json"


@dataclass(frozen=True)
class BackendCaps:
    binary: str
    version: str | None
    flags: frozenset[str]

    def supports(self, *aliases: str) -> bool:
        return any(alias in self.flags for alias in aliases)


def parse_help(text: str) -> tuple[frozenset[str], str | None]:
    flags: set[str] = set()
    version: str | None = None
    for line in text.splitlines():
        if not line or line[0].isspace():
            continue
        if "argument has been removed" in line.lower():
            continue
        parts = line.split()
        if not parts or not parts[0].startswith("-"):
            continue
        for index, token in enumerate(parts):
            if not token.startswith("-") or (index > 0 and not parts[index - 1].endswith(",")):
                break
            flag = token.rstrip(",")
            if flag != "-----":
                flags.add(flag)
    for line in text.splitlines():
        match = re.search(r"\bversion\b\s*[:=]?\s*(\S+)", line, re.IGNORECASE)
        if match and not line.lstrip().startswith("-"):
            version = line.strip()
            break
    return frozenset(flags), version


def _resolve_binary(binary: str) -> Path | None:
    candidate = Path(binary)
    if candidate.exists():
        return candidate.resolve()
    located = shutil.which(binary)
    return Path(located).resolve() if located else None


def _cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"


def _write_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def llamacpp_caps(
    binary: str = "llama-server", cache_path: Path | None = None
) -> BackendCaps | None:
    resolved = _resolve_binary(binary)
    if resolved is None:
        return None
    try:
        key = _cache_key(resolved)
    except OSError:
        return None
    target = cache_path or CAPS_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        cached = payload.get("entries", {}).get(key)
        if isinstance(cached, dict):
            return BackendCaps(
                str(cached["binary"]),
                str(cached["version"]) if cached.get("version") else None,
                frozenset(str(flag) for flag in cached.get("flags", [])),
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    try:
        result = subprocess.run(
            [str(resolved), "--help"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    flags, help_version = parse_help(
        (result.stdout or "") + "\n" + (result.stderr or "")
    )
    if not flags:
        return None
    version: str | None = help_version
    try:
        version_result = subprocess.run(
            [str(resolved), "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        version_text = (version_result.stdout or version_result.stderr or "").strip()
        if version_text:
            version = version_text.splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    caps = BackendCaps(str(resolved), version, flags)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    entries = payload.setdefault("entries", {})
    if isinstance(entries, dict):
        entries[key] = {
            "binary": caps.binary,
            "version": caps.version,
            "flags": sorted(caps.flags),
        }
        try:
            _write_cache(target, payload)
        except OSError:
            pass
    return caps
