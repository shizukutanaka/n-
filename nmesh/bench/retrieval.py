from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import httpx

from nmesh.paths import nmesh_home

RETRIEVAL_HARNESS_VERSION = "retrieval-v3"
RETRIEVAL_DOCS = 8
RETRIEVAL_SEEDS = (11, 23, 37, 51, 67, 79, 83, 97)
RETRIEVAL_RUNG_WORDS = (100, 400, 800, 1600, 2400, 3000)
RETRIEVAL_NEEDLE_POSITION = 0.7
RETRIEVAL_PASS_RATIO = 0.875
RETRIEVAL_FAIL_RATIO = 0.5
RETRIEVAL_GROWTH_RATIO = 1.2

_TOPIC = (
    "the archive maintenance rotation schedules storage racks and verifies "
    "checksum manifests for each catalogued reel before the quarterly audit "
    "review closes and the retention window advances one further cycle "
)
_NEEDLE = "The Meridian archive access protocol uses code {code} ."
_QUESTION = "What is the code for the Meridian archive access protocol?"


def measure_retrieval_estimate(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    *,
    encode_tps: float,
    seeds: Sequence[int] = RETRIEVAL_SEEDS,
    rung_words: Sequence[int] = RETRIEVAL_RUNG_WORDS,
) -> tuple[int, int]:
    """Return (requests, estimated seconds) for the retrieval ladder.

    One calibration request on a real retrieval document yields the tokens per
    ladder word for this tokenizer, so the estimate comes from the workload's
    own token volume rather than a fixed minute count: wall time scales with
    the host's measured encode throughput.
    """
    if not seeds or not rung_words:
        raise RuntimeError("retrieval seeds and rung words must not be empty")
    if encode_tps <= 0:
        raise RuntimeError("encode throughput must be positive")
    documents, _ = _documents(random.Random(0), RETRIEVAL_RUNG_WORDS[0], "000000")
    _, served = _embed(
        client, f"{base_url}/v1/embeddings", model_ref, documents[0]
    )
    if served <= 0:
        raise RuntimeError("retrieval estimate calibration served zero tokens")
    tokens_per_word = served / RETRIEVAL_RUNG_WORDS[0]
    requests = len(seeds) * len(rung_words) * (RETRIEVAL_DOCS + 1)
    words = sum(
        len(seeds) * (RETRIEVAL_DOCS * rung + len(_QUESTION.split()))
        for rung in rung_words
    )
    seconds = round(words * tokens_per_word / encode_tps)
    return requests, seconds


@dataclass(frozen=True)
class RetrievalRung:
    words: int
    served_tokens: int
    hits: int
    trials: int
    saturated: bool


@dataclass(frozen=True)
class RetrievalChunkArm:
    doc_words: int
    chunk_words: int
    chunk_tokens: int
    hits: int
    trials: int
    pool_hits: int
    pool_trials: int


@dataclass(frozen=True)
class RetrievalLimit:
    degraded_tokens: int
    chunk_tokens: int | None
    chunk_recovers: bool | None
    chunk_hits: int | None = None
    chunk_trials: int | None = None
    pool_recovers: bool | None = None
    pool_hits: int | None = None
    pool_trials: int | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    model_id: str
    quant: str
    backend: str
    gpu_name: str
    n_gpu_layers: int
    rungs: tuple[RetrievalRung, ...]
    digest: str
    harness: str
    at: float
    chunk: RetrievalChunkArm | None = None

    @property
    def control_passed(self) -> bool:
        return bool(
            self.rungs
            and self.rungs[0].hits == self.rungs[0].trials
            and not self.rungs[0].saturated
        )

    @property
    def usable_tokens(self) -> int | None:
        if not self.control_passed:
            return None
        usable: int | None = None
        for index, rung in enumerate(self.rungs):
            if rung.saturated or rung.hits / rung.trials < RETRIEVAL_PASS_RATIO:
                break
            if index > 0:
                usable = rung.served_tokens
        return usable

    @property
    def degraded_tokens(self) -> int | None:
        usable = self.usable_tokens
        if usable is None:
            return None
        for rung in self.rungs:
            if (
                not rung.saturated
                and rung.served_tokens > usable
                and rung.hits / rung.trials <= RETRIEVAL_FAIL_RATIO
            ):
                return rung.served_tokens
        return None

    @property
    def chunk_recovers(self) -> bool | None:
        if self.chunk is None or self.degraded_tokens is None:
            return None
        return self.chunk.hits / self.chunk.trials >= RETRIEVAL_PASS_RATIO

    @property
    def pool_recovers(self) -> bool | None:
        if self.chunk is None or self.degraded_tokens is None:
            return None
        return (
            self.chunk.pool_hits / self.chunk.pool_trials
            >= RETRIEVAL_PASS_RATIO
        )


def retrieval_key(
    model_id: str,
    quant: str,
    backend: str,
    gpu_name: str,
    n_gpu_layers: int,
) -> str:
    return f"{model_id}|{quant}|{backend}|{gpu_name}|{n_gpu_layers}"


def retrieval_digest(
    seeds: Sequence[int] = RETRIEVAL_SEEDS,
    rung_words: Sequence[int] = RETRIEVAL_RUNG_WORDS,
) -> str:
    definition = {
        "topic": _TOPIC,
        "needle": _NEEDLE,
        "question": _QUESTION,
        "docs": RETRIEVAL_DOCS,
        "position": RETRIEVAL_NEEDLE_POSITION,
        "rung_words": tuple(rung_words),
        "seeds": tuple(seeds),
    }
    encoded = json.dumps(
        definition, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _rung(data: object) -> RetrievalRung | None:
    if not isinstance(data, dict):
        return None
    try:
        words = data["words"]
        served_tokens = data["served_tokens"]
        hits = data["hits"]
        trials = data["trials"]
        saturated = data["saturated"]
        if (
            isinstance(words, bool)
            or not isinstance(words, int)
            or words < 0
            or isinstance(served_tokens, bool)
            or not isinstance(served_tokens, int)
            or served_tokens < 0
            or isinstance(hits, bool)
            or not isinstance(hits, int)
            or hits < 0
            or isinstance(trials, bool)
            or not isinstance(trials, int)
            or trials < 1
            or hits > trials
            or not isinstance(saturated, bool)
        ):
            return None
        return RetrievalRung(words, served_tokens, hits, trials, saturated)
    except (KeyError, TypeError, ValueError):
        return None


def _chunk(data: object) -> RetrievalChunkArm | None:
    if not isinstance(data, dict):
        return None
    try:
        doc_words = data["doc_words"]
        chunk_words = data["chunk_words"]
        chunk_tokens = data["chunk_tokens"]
        hits = data["hits"]
        trials = data["trials"]
        pool_hits = data["pool_hits"]
        pool_trials = data["pool_trials"]
        if (
            isinstance(doc_words, bool)
            or not isinstance(doc_words, int)
            or doc_words < 1
            or isinstance(chunk_words, bool)
            or not isinstance(chunk_words, int)
            or chunk_words < 1
            or isinstance(chunk_tokens, bool)
            or not isinstance(chunk_tokens, int)
            or chunk_tokens < 1
            or isinstance(hits, bool)
            or not isinstance(hits, int)
            or hits < 0
            or isinstance(trials, bool)
            or not isinstance(trials, int)
            or trials < 1
            or hits > trials
            or isinstance(pool_hits, bool)
            or not isinstance(pool_hits, int)
            or pool_hits < 0
            or isinstance(pool_trials, bool)
            or not isinstance(pool_trials, int)
            or pool_trials < 1
            or pool_hits > pool_trials
        ):
            return None
        return RetrievalChunkArm(
            doc_words, chunk_words, chunk_tokens, hits, trials,
            pool_hits, pool_trials,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _record(data: object) -> RetrievalRecord | None:
    if not isinstance(data, dict):
        return None
    try:
        model_id = data["model_id"]
        quant = data["quant"]
        backend = data["backend"]
        gpu_name = data["gpu_name"]
        n_gpu_layers = data["n_gpu_layers"]
        rungs_data = data["rungs"]
        digest = data["digest"]
        harness = data["harness"]
        at = data["at"]
        chunk_data = data.get("chunk")
        if (
            not isinstance(model_id, str)
            or not isinstance(quant, str)
            or not isinstance(backend, str)
            or not isinstance(gpu_name, str)
            or isinstance(n_gpu_layers, bool)
            or not isinstance(n_gpu_layers, int)
            or n_gpu_layers < 0
            or not isinstance(rungs_data, list)
            or not isinstance(digest, str)
            or len(digest) != 16
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(harness, str)
            or isinstance(at, bool)
            or not isinstance(at, (int, float))
            or not math.isfinite(at)
            or at < 0
        ):
            return None
        rungs = tuple(_rung(value) for value in rungs_data)
        if not rungs or any(value is None for value in rungs):
            return None
        rung_values = tuple(value for value in rungs if value is not None)
        if any(
            previous.words >= current.words
            for previous, current in pairwise(rung_values)
        ):
            return None
        chunk = None
        if chunk_data is not None:
            chunk = _chunk(chunk_data)
            if chunk is None:
                return None
        return RetrievalRecord(
            model_id,
            quant,
            backend,
            gpu_name,
            n_gpu_layers,
            rung_values,
            digest,
            harness,
            float(at),
            chunk,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_retrieval_cache(path: Path | None = None) -> dict[str, RetrievalRecord]:
    target = path or (nmesh_home() / "retrieval.json")
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


def save_retrieval(
    record: RetrievalRecord, path: Path | None = None
) -> Path:
    target = path or (nmesh_home() / "retrieval.json")
    records = load_retrieval_cache(target)
    key = retrieval_key(
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


def _documents(
    rng: random.Random, words: int, code: str
) -> tuple[list[str], int]:
    pool = _TOPIC.split()
    target = rng.randrange(RETRIEVAL_DOCS)
    at = int(words * RETRIEVAL_NEEDLE_POSITION)
    needle = _NEEDLE.format(code=code).split()
    documents: list[str] = []
    for index in range(RETRIEVAL_DOCS):
        body = [rng.choice(pool) for _ in range(words)]
        if index == target:
            body = body[:at] + needle + body[at:]
        documents.append(" ".join(body))
    return documents, target


def _embedding_payload(response: httpx.Response) -> tuple[list[list[float]], int]:
    response.raise_for_status()
    try:
        payload: object = response.json()
    except (TypeError, ValueError) as error:
        raise RuntimeError("retrieval response was not valid JSON") from error
    usage = payload.get("usage") if isinstance(payload, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise RuntimeError(
            "retrieval response did not report integer usage.prompt_tokens"
        )
    values = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise RuntimeError("retrieval response did not report embedding data")
    vectors: list[list[float]] = []
    for value in values:
        vector = value.get("embedding") if isinstance(value, dict) else None
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in vector
            )
        ):
            raise RuntimeError("retrieval response contained invalid embedding data")
        vectors.append([float(item) for item in vector])
    return vectors, tokens


def _embed(
    client: httpx.Client, url: str, model_ref: str, input_value: str
) -> tuple[list[float], int]:
    vectors, tokens = _embedding_payload(
        client.post(url, json={"model": model_ref, "input": input_value})
    )
    if len(vectors) != 1:
        raise RuntimeError("retrieval response returned an unexpected vector count")
    return vectors[0], tokens


def _embed_inputs(
    client: httpx.Client,
    url: str,
    model_ref: str,
    inputs: list[str],
) -> list[list[float]]:
    vectors, _ = _embedding_payload(
        client.post(url, json={"model": model_ref, "input": inputs})
    )
    if len(vectors) != len(inputs):
        raise RuntimeError("retrieval response returned an unexpected vector count")
    return vectors


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def pool_embeddings(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot pool an empty vector list")
    dimensions = len(vectors[0])
    if not dimensions or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("cannot pool vectors with different dimensions")
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        normalized.append(
            [value / norm for value in vector] if norm else list(vector)
        )
    return [
        sum(vector[index] for vector in normalized) / len(normalized)
        for index in range(dimensions)
    ]


def measure_retrieval_chunk_arm(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    *,
    doc_words: int,
    chunk_words: int,
    chunk_tokens: int,
    seeds: Sequence[int] = RETRIEVAL_SEEDS,
) -> RetrievalChunkArm:
    if (
        isinstance(doc_words, bool)
        or not isinstance(doc_words, int)
        or doc_words < 1
        or isinstance(chunk_words, bool)
        or not isinstance(chunk_words, int)
        or chunk_words < 1
        or isinstance(chunk_tokens, bool)
        or not isinstance(chunk_tokens, int)
        or chunk_tokens < 1
    ):
        raise RuntimeError(
            "retrieval document, chunk, and token counts must be positive"
        )
    if not seeds:
        raise RuntimeError("retrieval seeds must not be empty")
    url = f"{base_url}/v1/embeddings"
    hits = 0
    pool_hits = 0
    for seed in seeds:
        rng = random.Random(seed)
        code = f"{rng.randrange(16**6):06X}"
        documents, target = _documents(rng, doc_words, code)
        document_vectors: list[list[list[float]]] = []
        for document in documents:
            words = document.split()
            chunks = [
                " ".join(words[index:index + chunk_words])
                for index in range(0, len(words), chunk_words)
            ]
            document_vectors.append(
                _embed_inputs(client, url, model_ref, chunks)
            )
        query, _ = _embed(client, url, model_ref, _QUESTION)
        scores = [
            max(_cosine(query, vector) for vector in vectors)
            for vectors in document_vectors
        ]
        if max(range(RETRIEVAL_DOCS), key=scores.__getitem__) == target:
            hits += 1
        pooled_scores = [
            _cosine(query, pool_embeddings(vectors))
            for vectors in document_vectors
        ]
        if max(range(RETRIEVAL_DOCS), key=pooled_scores.__getitem__) == target:
            pool_hits += 1
    return RetrievalChunkArm(
        doc_words=doc_words,
        chunk_words=chunk_words,
        chunk_tokens=chunk_tokens,
        hits=hits,
        trials=len(seeds),
        pool_hits=pool_hits,
        pool_trials=len(seeds),
    )


def measure_retrieval(
    client: httpx.Client,
    base_url: str,
    model_ref: str,
    *,
    seeds: Sequence[int] = RETRIEVAL_SEEDS,
    rung_words: Sequence[int] = RETRIEVAL_RUNG_WORDS,
    cap: int | None = None,
) -> tuple[RetrievalRung, ...]:
    if not seeds or not rung_words:
        raise RuntimeError("retrieval seeds and rung words must not be empty")
    if cap is not None and cap < 0:
        raise RuntimeError("retrieval cap must not be negative")
    url = f"{base_url}/v1/embeddings"
    rungs: list[RetrievalRung] = []
    for words in rung_words:
        if words < 1:
            raise RuntimeError("retrieval rung words must be positive")
        hits = 0
        target_tokens: list[int] = []
        for seed in seeds:
            rng = random.Random(seed)
            code = f"{rng.randrange(16**6):06X}"
            documents, target = _documents(rng, words, code)
            # Independent document embeds are issued concurrently
            # (order preserved); measured quality is unchanged.
            with ThreadPoolExecutor(max_workers=len(documents)) as pool:
                results = list(pool.map(
                    lambda document: _embed(client, url, model_ref, document),
                    documents,
                ))
            vectors = [vector for vector, _ in results]
            target_tokens.append(results[target][1])
            query, _ = _embed(client, url, model_ref, _QUESTION)
            scores = [_cosine(query, vector) for vector in vectors]
            order = sorted(
                range(RETRIEVAL_DOCS),
                key=lambda index: scores[index],
                reverse=True,
            )
            if order[0] == target:
                hits += 1
        served_tokens = round(statistics.median(target_tokens))
        saturated = (
            cap is not None and served_tokens >= cap - 2
        ) or (
            bool(rungs)
            and served_tokens < RETRIEVAL_GROWTH_RATIO
            * rungs[-1].served_tokens
        )
        rung = RetrievalRung(
            words, served_tokens, hits, len(seeds), saturated
        )
        rungs.append(rung)
        if len(rungs) >= 2 and rungs[-1].saturated and rungs[-2].saturated:
            break
    return tuple(rungs)
