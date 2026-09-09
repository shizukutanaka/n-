from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nmesh.paths import nmesh_home

if TYPE_CHECKING:
    from httpx import AsyncClient

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


def calibration_key(model_id: str, chat: bool) -> str:
    return f"{model_id}|{'chat' if chat else 'text'}"


@dataclass(frozen=True)
class Calibration:
    cjk_per_char: float
    other_per_char: float
    samples: int
    measured: bool
    overhead: float = 0.0


@dataclass(frozen=True)
class Sums:
    n: int = 0
    s_cc: float = 0.0
    s_co: float = 0.0
    s_oo: float = 0.0
    s_ct: float = 0.0
    s_ot: float = 0.0
    s_c: float = 0.0
    s_o: float = 0.0
    s_t: float = 0.0


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
        cjk * selected.cjk_per_char
        + other * selected.other_per_char
        + selected.overhead
    ))


def fit(sums: Sums) -> Calibration:
    if sums.n < MIN_SAMPLES:
        return _defaults(sums.n)
    a11, a12, a13 = sums.s_cc, sums.s_co, sums.s_c
    a22, a23 = sums.s_oo, sums.s_o
    a33 = sums.n
    b1, b2, b3 = sums.s_ct, sums.s_ot, sums.s_t
    determinant = (
        a11 * (a22 * a33 - a23 * a23)
        - a12 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * a23 - a22 * a13)
    )
    if abs(determinant) < 1e-9:
        return _defaults(sums.n)
    cjk = (
        b1 * (a22 * a33 - a23 * a23)
        - a12 * (b2 * a33 - a23 * b3)
        + a13 * (b2 * a23 - a22 * b3)
    ) / determinant
    other = (
        a11 * (b2 * a33 - a23 * b3)
        - b1 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * b3 - b2 * a13)
    ) / determinant
    overhead = (
        a11 * (a22 * b3 - b2 * a23)
        - a12 * (a12 * b3 - b2 * a13)
        + b1 * (a12 * a23 - a22 * a13)
    ) / determinant
    clamped_cjk = min(2.0, max(0.05, cjk))
    clamped_other = min(2.0, max(0.05, other))
    clamped_overhead = min(512.0, max(0.0, overhead))
    if (
        clamped_cjk != cjk
        or clamped_other != other
        or clamped_overhead != overhead
    ):
        return _defaults(sums.n)
    return Calibration(
        clamped_cjk, clamped_other, sums.n, True, clamped_overhead
    )


def _path() -> Path:
    return nmesh_home() / "tokens.json"


def _coerce_sums(value: object) -> Sums:
    if not isinstance(value, dict):
        return Sums()
    required = {
        "n", "s_cc", "s_co", "s_oo", "s_ct", "s_ot", "s_c", "s_o", "s_t",
    }
    if not required.issubset(value):
        return Sums()
    try:
        return Sums(
            n=max(0, int(value.get("n", 0))),
            s_cc=float(value.get("s_cc", 0.0)),
            s_co=float(value.get("s_co", 0.0)),
            s_oo=float(value.get("s_oo", 0.0)),
            s_ct=float(value.get("s_ct", 0.0)),
            s_ot=float(value.get("s_ot", 0.0)),
            s_c=float(value.get("s_c", 0.0)),
            s_o=float(value.get("s_o", 0.0)),
            s_t=float(value.get("s_t", 0.0)),
        )
    except (TypeError, ValueError):
        return Sums()


def _read() -> dict[str, Sums]:
    try:
        payload = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        return {}
    return {
        str(name): _coerce_sums(value)
        for name, value in payload["models"].items()
    }


def _write(values: dict[str, Sums]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps({"models": {
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


def all_sums() -> dict[str, Sums]:
    with _LOCK:
        return _read()


def load_sums(model_id: str) -> Sums:
    with _LOCK:
        return _read().get(model_id, Sums())


def calibration_for(model_id: str) -> Calibration:
    return fit(load_sums(model_id))


def record(model_id: str, text: str, exact_tokens: int) -> None:
    if exact_tokens < 0:
        return
    cjk, other = split_chars(text)
    with _LOCK:
        values = _read()
        previous = values.get(model_id, Sums())
        values[model_id] = Sums(
            n=previous.n + 1,
            s_cc=previous.s_cc + cjk * cjk,
            s_co=previous.s_co + cjk * other,
            s_oo=previous.s_oo + other * other,
            s_ct=previous.s_ct + cjk * exact_tokens,
            s_ot=previous.s_ot + other * exact_tokens,
            s_c=previous.s_c + cjk,
            s_o=previous.s_o + other,
            s_t=previous.s_t + exact_tokens,
        )
        _write(values)


async def exact_tokens(base_url: str, text: str, client: AsyncClient) -> int | None:
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
    # Any client or transport error must fall back to the estimate, never fail the request.
    except Exception:  # noqa: BLE001
        return None
    return None
