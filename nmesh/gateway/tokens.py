from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nmesh.paths import nmesh_home

CJK_RANGES = (
    (0x3000, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xAC00, 0xD7AF),
    (0xFF00, 0xFFEF),
)
DEFAULT_CJK_PER_CHAR = 1.0
DEFAULT_OTHER_PER_CHAR = 0.25
MIN_SAMPLES = 20


@dataclass(frozen=True)
class Calibration:
    cjk_per_char: float
    other_per_char: float
    samples: int
    measured: bool


@dataclass(frozen=True)
class Sums:
    n: int = 0
    s_cc: float = 0.0
    s_co: float = 0.0
    s_oo: float = 0.0
    s_ct: float = 0.0
    s_ot: float = 0.0


_LOCK = threading.Lock()


def _defaults(samples: int = 0) -> Calibration:
    return Calibration(
        DEFAULT_CJK_PER_CHAR,
        DEFAULT_OTHER_PER_CHAR,
        samples,
        False,
    )


def split_chars(text: str) -> tuple[int, int]:
    cjk = sum(
        any(start <= ord(char) <= end for start, end in CJK_RANGES)
        for char in text
    )
    return cjk, len(text) - cjk


def estimate_tokens(text: str, calibration: Calibration | None = None) -> int:
    if not text:
        return 0
    selected = calibration or _defaults()
    cjk, other = split_chars(text)
    return max(1, math.ceil(
        cjk * selected.cjk_per_char + other * selected.other_per_char
    ))


def fit(sums: Sums) -> Calibration:
    if sums.n < MIN_SAMPLES:
        return _defaults(sums.n)
    determinant = sums.s_cc * sums.s_oo - sums.s_co * sums.s_co
    if abs(determinant) < 1e-9:
        return _defaults(sums.n)
    cjk = (sums.s_ct * sums.s_oo - sums.s_ot * sums.s_co) / determinant
    other = (sums.s_ot * sums.s_cc - sums.s_ct * sums.s_co) / determinant
    clamped_cjk = min(2.0, max(0.05, cjk))
    clamped_other = min(2.0, max(0.05, other))
    if clamped_cjk != cjk or clamped_other != other:
        return _defaults(sums.n)
    return Calibration(clamped_cjk, clamped_other, sums.n, True)


def _path() -> Path:
    return nmesh_home() / "tokens.json"


def _coerce_sums(value: object) -> Sums:
    if not isinstance(value, dict):
        return Sums()
    try:
        return Sums(
            n=max(0, int(value.get("n", 0))),
            s_cc=float(value.get("s_cc", 0.0)),
            s_co=float(value.get("s_co", 0.0)),
            s_oo=float(value.get("s_oo", 0.0)),
            s_ct=float(value.get("s_ct", 0.0)),
            s_ot=float(value.get("s_ot", 0.0)),
        )
    except (TypeError, ValueError):
        return Sums()


def _read() -> dict[str, Sums]:
    try:
        payload = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("services"), dict):
        return {}
    return {
        str(name): _coerce_sums(value)
        for name, value in payload["services"].items()
    }


def _write(values: dict[str, Sums]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps({"services": {
                name: asdict(sums) for name, sums in values.items()
            }}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def load_sums(service: str) -> Sums:
    with _LOCK:
        return _read().get(service, Sums())


def calibration_for(service: str) -> Calibration:
    return fit(load_sums(service))


def record(service: str, text: str, exact_tokens: int) -> None:
    if exact_tokens < 0:
        return
    cjk, other = split_chars(text)
    with _LOCK:
        values = _read()
        previous = values.get(service, Sums())
        values[service] = Sums(
            n=previous.n + 1,
            s_cc=previous.s_cc + cjk * cjk,
            s_co=previous.s_co + cjk * other,
            s_oo=previous.s_oo + other * other,
            s_ct=previous.s_ct + cjk * exact_tokens,
            s_ot=previous.s_ot + other * exact_tokens,
        )
        _write(values)


async def exact_tokens(base_url: str, text: str, client: Any) -> int | None:
    # vLLM's tokenize endpoint is intentionally unsupported until its shape is verified.
    try:
        response = await client.post(
            f"{base_url}/tokenize",
            json={"content": text},
            timeout=0.5,
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
        if isinstance(tokens, int) and tokens >= 0:
            return tokens
    except Exception:  # noqa: BLE001
        return None
    return None
