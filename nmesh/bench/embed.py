from __future__ import annotations

import json
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from nmesh.paths import nmesh_home

EMBED_HARNESS_VERSION = "embed-v1"
EMBED_CAP_TRUNCATION_RATIO = 0.9
EMBED_CAP_AGREEMENT_TOKENS = 2
EMBED_CAP_AGREEMENT_RATIO = 0.01
EMBED_ENCODE_WORDS = 512


@dataclass(frozen=True)
class EmbedRecord:
    model_id: str
    quant: str
    backend: str
    gpu_name: str
    n_gpu_layers: int
    requested_context: int
    probe_tokens_small: int
    served_small: int
    probe_tokens_large: int
    served_large: int
    encode_tps: float
    encode_tps_min: float
    encode_tps_max: float
    encode_input_tokens: int
    runs: int
    harness: str
    at: float
    refused_small: bool = False
    refused_large: bool = False

    @property
    def cap(self) -> int | None:
        """Return a conservative cap only when both probes were truncated.

        The served counts may differ by a token or two because tokenization is
        not perfectly stable across probe sizes. Requiring both probes to be
        substantially shorter than requested, then taking the smaller served
        count within a small agreement tolerance, prevents a silently
        truncated input from exceeding the proven ceiling.

        A probe the backend refused outright proves the opposite of silent
        truncation, so no cap follows from it.
        """
        if self.refused_small or self.refused_large:
            return None
        if not (
            self.served_small
            < self.probe_tokens_small * EMBED_CAP_TRUNCATION_RATIO
            and self.served_large
            < self.probe_tokens_large * EMBED_CAP_TRUNCATION_RATIO
        ):
            return None
        difference = abs(self.served_small - self.served_large)
        tolerance = max(
            EMBED_CAP_AGREEMENT_TOKENS,
            EMBED_CAP_AGREEMENT_RATIO
            * min(self.served_small, self.served_large),
        )
        if difference > tolerance:
            return None
        return min(self.served_small, self.served_large)


@dataclass(frozen=True)
class EmbedMeasurement:
    probe_tokens_small: int
    served_small: int
    probe_tokens_large: int
    served_large: int
    encode_tps: float
    encode_tps_min: float
    encode_tps_max: float
    encode_input_tokens: int
    runs: int
    refused_small: bool = False
    refused_large: bool = False


def embed_key(
    model_id: str,
    quant: str,
    backend: str,
    gpu_name: str,
    n_gpu_layers: int,
) -> str:
    return f"{model_id}|{quant}|{backend}|{gpu_name}|{n_gpu_layers}"


def _record(data: object) -> EmbedRecord | None:
    if not isinstance(data, dict):
        return None
    try:
        model_id = data["model_id"]
        quant = data["quant"]
        backend = data["backend"]
        gpu_name = data["gpu_name"]
        harness = data["harness"]
        n_gpu_layers = data["n_gpu_layers"]
        requested_context = data["requested_context"]
        probe_tokens_small = data["probe_tokens_small"]
        served_small = data["served_small"]
        probe_tokens_large = data["probe_tokens_large"]
        served_large = data["served_large"]
        encode_tps = data["encode_tps"]
        encode_tps_min = data["encode_tps_min"]
        encode_tps_max = data["encode_tps_max"]
        encode_input_tokens = data["encode_input_tokens"]
        runs = data["runs"]
        at = data["at"]
        refused_small = data.get("refused_small", False)
        refused_large = data.get("refused_large", False)
        if (
            not isinstance(refused_small, bool)
            or not isinstance(refused_large, bool)
        ):
            return None
        if (
            not isinstance(model_id, str)
            or not isinstance(quant, str)
            or not isinstance(backend, str)
            or not isinstance(gpu_name, str)
            or not isinstance(harness, str)
            or isinstance(n_gpu_layers, bool)
            or not isinstance(n_gpu_layers, int)
            or n_gpu_layers < 0
            or isinstance(requested_context, bool)
            or not isinstance(requested_context, int)
            or requested_context < 0
            or isinstance(probe_tokens_small, bool)
            or not isinstance(probe_tokens_small, int)
            or probe_tokens_small < 0
            or isinstance(served_small, bool)
            or not isinstance(served_small, int)
            or served_small < 0
            or isinstance(probe_tokens_large, bool)
            or not isinstance(probe_tokens_large, int)
            or probe_tokens_large < 0
            or isinstance(served_large, bool)
            or not isinstance(served_large, int)
            or served_large < 0
            or isinstance(encode_input_tokens, bool)
            or not isinstance(encode_input_tokens, int)
            or encode_input_tokens < 0
            or isinstance(runs, bool)
            or not isinstance(runs, int)
            or runs < 1
            or isinstance(encode_tps, bool)
            or not isinstance(encode_tps, (int, float))
            or not math.isfinite(encode_tps)
            or encode_tps < 0
            or isinstance(encode_tps_min, bool)
            or not isinstance(encode_tps_min, (int, float))
            or not math.isfinite(encode_tps_min)
            or encode_tps_min < 0
            or isinstance(encode_tps_max, bool)
            or not isinstance(encode_tps_max, (int, float))
            or not math.isfinite(encode_tps_max)
            or encode_tps_max < 0
            or isinstance(at, bool)
            or not isinstance(at, (int, float))
            or not math.isfinite(at)
            or at < 0
        ):
            return None
        return EmbedRecord(
            model_id,
            quant,
            backend,
            gpu_name,
            n_gpu_layers,
            requested_context,
            probe_tokens_small,
            served_small,
            probe_tokens_large,
            served_large,
            float(encode_tps),
            float(encode_tps_min),
            float(encode_tps_max),
            encode_input_tokens,
            runs,
            harness,
            float(at),
            refused_small,
            refused_large,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_embed_cache(path: Path | None = None) -> dict[str, EmbedRecord]:
    target = path or (nmesh_home() / "embed.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict):
            return {}
        return {
            str(key): record
            for key, value in results.items()
            if (record := _record(value)) is not None
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_embed(record: EmbedRecord, path: Path | None = None) -> Path:
    target = path or (nmesh_home() / "embed.json")
    records = load_embed_cache(target)
    key = embed_key(
        record.model_id,
        record.quant,
        record.backend,
        record.gpu_name,
        record.n_gpu_layers,
    )
    records[key] = record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"results": {key: asdict(value) for key, value in records.items()}},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


_FILLER = "embedding filler text "


def _prompt(words: int) -> str:
    return f"{uuid.uuid4().hex} " + _FILLER * max(1, words)


def _usage_tokens(response: httpx.Response) -> int:
    response.raise_for_status()
    try:
        payload: object = response.json()
    except (TypeError, ValueError) as error:
        raise RuntimeError("embedding response was not valid JSON") from error
    usage = payload.get("usage") if isinstance(payload, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise RuntimeError("embedding response did not report integer usage.prompt_tokens")
    return tokens


def _measure_input(
    client: httpx.Client,
    url: str,
    model_ref: str,
    words: int,
) -> tuple[int, float]:
    started = time.perf_counter()
    response = client.post(
        url,
        json={"model": model_ref, "input": _prompt(words)},
    )
    tokens = _usage_tokens(response)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return tokens, elapsed


def _probe_input(
    client: httpx.Client,
    url: str,
    model_ref: str,
    words: int,
) -> int | None:
    """Return served tokens, or None when the backend refuses the input.

    An oversize probe is expected to be refused by backends that enforce a
    batch limit instead of truncating, and that refusal is the measurement
    rather than a failure.
    """
    try:
        tokens, _ = _measure_input(client, url, model_ref, words)
    except httpx.HTTPStatusError:
        return None
    return tokens


def measure_embedding(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    *,
    requested_context: int,
    runs: int = 3,
) -> EmbedMeasurement:
    if requested_context < 1:
        raise RuntimeError("requested embedding context must be positive")
    if runs < 1:
        raise RuntimeError("embedding runs must be positive")
    url = f"{base_url}/v1/embeddings"
    calibration_tokens, _ = _measure_input(client, url, model_ref, 256)
    if calibration_tokens <= 0:
        raise RuntimeError("embedding calibration reported zero prompt tokens")
    tokens_per_word = calibration_tokens / 256

    probe_tokens_small = max(1, round(requested_context * 1.5))
    probe_tokens_large = max(1, round(requested_context * 3.0))
    small_words = max(1, math.ceil(probe_tokens_small / tokens_per_word))
    large_words = max(1, math.ceil(probe_tokens_large / tokens_per_word))
    probed_small = _probe_input(client, url, model_ref, small_words)
    probed_large = _probe_input(client, url, model_ref, large_words)

    rates: list[float] = []
    served_mid: list[int] = []
    for _ in range(runs):
        served, elapsed = _measure_input(
            client, url, model_ref, EMBED_ENCODE_WORDS
        )
        served_mid.append(served)
        rates.append(served / elapsed)
    return EmbedMeasurement(
        probe_tokens_small,
        probed_small or 0,
        probe_tokens_large,
        probed_large or 0,
        statistics.median(rates),
        min(rates),
        max(rates),
        round(statistics.median(served_mid)),
        runs,
        probed_small is None,
        probed_large is None,
    )
