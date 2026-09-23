from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from nmesh import i18n
from nmesh.artifact import ollama_base_url, service_fingerprint
from nmesh.bench import benchmark_key
from nmesh.bench.embed import EMBED_HARNESS_VERSION, load_embed_cache
from nmesh.bench.retrieval import (
    RETRIEVAL_HARNESS_VERSION,
    load_retrieval_cache,
    pool_embeddings,
    retrieval_digest,
)
from nmesh.orchestrate import (
    PROTOCOL_VERSION,
    Delegation,
    DelegationRecord,
    Endpoint,
    Ledger,
    RoleIdentity,
    best_for,
    decide,
    delegate,
    load_cache,
)
from nmesh.planner import PLAN_PATH, Plan, PlannedService, load_plan
from nmesh.runtime import ensure_running, heartbeat, idle_services, unload
from nmesh.runtime import status as runtime_status
from nmesh.runtime.logs import log_path
from nmesh.runtime.logs import tail as tail_log
from nmesh.telemetry import Sample, summary_by_approximate
from nmesh.telemetry import record as record_telemetry
from nmesh.telemetry import summary as telemetry_summary

from .gate import SwapGate
from .jobs import Job, JobRegistry
from .limit import SlotLimiter
from .tokens import (
    Sums,
    all_sums,
    calibration_for,
    calibration_key,
    estimate_tokens,
    exact_tokens,
    fit,
)
from .tokens import (
    record as record_token_calibration,
)

try:
    QUEUE_TIMEOUT = float(os.environ.get("NMESH_QUEUE_TIMEOUT", "120.0"))
except ValueError:
    QUEUE_TIMEOUT = 120.0

try:
    KEEP_ALIVE = float(os.environ.get("NMESH_KEEP_ALIVE", "0"))
except ValueError:
    KEEP_ALIVE = 0.0

try:
    CONNECT_TIMEOUT = float(os.environ.get("NMESH_CONNECT_TIMEOUT", "10.0"))
except ValueError:
    CONNECT_TIMEOUT = 10.0

if TYPE_CHECKING:
    import httpx
    from fastapi import FastAPI, HTTPException
    from fastapi import FastAPI as FastAPIApp
    from fastapi.responses import Response, StreamingResponse
    from starlette.requests import Request
else:
    try:
        import httpx
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
        from starlette.requests import Request
    except ImportError:
        FastAPI = None
        httpx = None
        HTTPException = RuntimeError
        Response = None
        StreamingResponse = None
        Request = object


def _get(request: Mapping[str, object], key: str, default: object = None) -> object:
    return request.get(key, default)


def _upstream_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _upstream_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class _Ticket:
    __slots__ = ("peak",)

    def __init__(self) -> None:
        self.peak = 1


class _InFlight:
    def __init__(self) -> None:
        self._active: dict[str, set[_Ticket]] = {}

    def enter(self, service: str) -> _Ticket:
        ticket = _Ticket()
        active = self._active.setdefault(service, set())
        active.add(ticket)
        current = len(active)
        for item in active:
            item.peak = max(item.peak, current)
        return ticket

    def leave(self, service: str, ticket: _Ticket) -> int:
        active = self._active.get(service)
        if active is not None:
            active.discard(ticket)
            if not active:
                self._active.pop(service, None)
        return ticket.peak

    def count(self, service: str) -> int:
        return len(self._active.get(service, ()))


class _LastUse:
    def __init__(self, services: Sequence[PlannedService]) -> None:
        stamp = time.monotonic()
        self._values = {service.name: stamp for service in services}

    def touch(self, service: str, now: float | None = None) -> None:
        self._values[service] = time.monotonic() if now is None else now

    def age(self, service: str, now: float | None = None) -> float | None:
        last = self._values.get(service)
        if last is None:
            return None
        current = time.monotonic() if now is None else now
        return max(0.0, current - last)

    def ensure(self, services: Sequence[PlannedService]) -> None:
        stamp = time.monotonic()
        for service in services:
            self._values.setdefault(service.name, stamp)


def _timing_metrics(timings: object) -> tuple[float | None, float | None]:
    if not isinstance(timings, dict):
        return None, None
    predicted_n = _upstream_int(timings.get("predicted_n"))
    predicted_ms = _upstream_float(timings.get("predicted_ms"))
    decode = (
        predicted_n / (predicted_ms / 1000)
        if predicted_n is not None and predicted_n >= 1
        and predicted_ms is not None and predicted_ms > 0 else None
    )
    cache_n = _upstream_int(timings.get("cache_n"))
    prompt_n = _upstream_int(timings.get("prompt_n"))
    prompt_ms = _upstream_float(timings.get("prompt_ms"))
    # Partial cache leaves enough prompt_n/prompt_ms for an exact prefill rate.
    cached = max(cache_n or 0, 0)
    prefill = (
        prompt_n / (prompt_ms / 1000)
        if prompt_n is not None and prompt_n >= 16 and cached < prompt_n
        and prompt_ms is not None and prompt_ms > 0 else None
    )
    return decode, prefill


def _prompt_depth(usage: object, timings: object) -> int | None:
    if isinstance(usage, Mapping):
        prompt_tokens = _upstream_int(usage.get("prompt_tokens"))
        if prompt_tokens is not None:
            return prompt_tokens if prompt_tokens >= 0 else None
    if not isinstance(timings, Mapping):
        return None
    prompt_n = _upstream_int(timings.get("prompt_n"))
    cache_n = _upstream_int(timings.get("cache_n"))
    if prompt_n is None:
        return None
    if prompt_n < 0 or (cache_n is not None and cache_n < 0):
        return None
    # Cached prefill still counts: decode runs at the KV position set by the
    # whole prompt, whether those tokens were recomputed or reused.
    return prompt_n + (cache_n or 0)


def _is_chat_request(request: Mapping[str, object]) -> bool:
    return "messages" in request


def _unload_reason(
    service: str,
    item: Mapping[str, object] | None,
    idle: set[str],
) -> str:
    if item is not None and item.get("shared"):
        return "shared"
    if item is not None and item.get("external"):
        return "external"
    if service in idle or (item is not None and item.get("idle")):
        return "idle"
    if item is None or not item.get("running", False):
        return "not_running"
    return "not_owned"


async def _unload_service(
    service: str,
    item: Mapping[str, object] | None,
    idle: set[str],
) -> dict[str, object]:
    if await asyncio.to_thread(unload, service):
        return {"service": service, "unloaded": True, "reason": "ok"}
    return {
        "service": service,
        "unloaded": False,
        "reason": _unload_reason(service, item, idle),
    }


def _content(request: Mapping[str, object]) -> str:
    messages = _get(request, "messages", [])
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for item in messages:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    part.get("text", "")
                    for part in content
                    if (
                        isinstance(part, Mapping)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    )
                )
        return " ".join(parts)
    prompt = _get(request, "prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return " ".join(item for item in prompt if isinstance(item, str))
    return ""


def route(
    request: Mapping[str, object],
    plan: Plan,
    token_hint: int | None = None,
) -> str:
    explicit = _explicit(_get(request, "model"), plan)
    if explicit is not None:
        return explicit
    if _get(request, "tools"):
        return (
            plan.routing.role_to_service.get("tool")
            or plan.routing.role_to_service.get("tools")
            or plan.routing.role_to_service.get("chat", "")
        )
    content = _content(request)
    if re.search(r"(?:```|^\s*(?:def |class |function |SELECT |import ))", content, re.MULTILINE):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    chat_name = plan.routing.role_to_service.get("chat", "")
    chat = next((item for item in plan.services if item.name == chat_name), None)
    reserved = _reserved_tokens(request)
    calibration = (
        calibration_for(calibration_key(chat.model_id, _is_chat_request(request)))
        if chat is not None
        else None
    )
    token_count = (
        estimate_tokens(content, calibration) if token_hint is None else token_hint
    )
    if chat and token_count + reserved > chat.context * 0.8:
        return max(plan.services, key=lambda item: item.context, default=chat).name
    return chat_name or (plan.services[0].name if plan.services else "")


def _explicit(model: object, plan: Plan) -> str | None:
    if not isinstance(model, str) or model in {"", "nmesh-auto"}:
        return None
    name = model.removeprefix("nmesh-")
    if any(item.name == name for item in plan.services):
        return name
    return plan.routing.role_to_service.get(name) or None


def _service(plan: Plan, name: str) -> PlannedService:
    service = next((item for item in plan.services if item.name == name), None)
    if service is None:
        raise HTTPException(status_code=404, detail=f"Unknown service: {name}")
    return service


def _embed_input_caps(plan: Plan) -> dict[tuple[str, str, str], int]:
    services = [service for service in plan.services if service.roles == ["embed"]]
    if not services:
        return {}
    records = load_embed_cache()
    caps: dict[tuple[str, str, str], int] = {}
    for service in services:
        key = (
            service.model_id.casefold(),
            service.quant.casefold(),
            service.backend.casefold(),
        )
        newest = max(
            (
                record
                for record in records.values()
                if record.harness == EMBED_HARNESS_VERSION
                and record.model_id.casefold() == key[0]
                and record.quant.casefold() == key[1]
                and record.backend.casefold() == key[2]
                and record.cap is not None
            ),
            key=lambda record: record.at,
            default=None,
        )
        if newest is not None and newest.cap is not None:
            caps[key] = newest.cap
    return caps


@dataclass(frozen=True)
class EmbedChunkPlan:
    chunk_words: int
    chunk_tokens: int


def _embed_chunk_plans(
    plan: Plan,
) -> dict[tuple[str, str, str], EmbedChunkPlan]:
    services = [service for service in plan.services if service.roles == ["embed"]]
    if not services:
        return {}
    records = load_retrieval_cache()
    plans: dict[tuple[str, str, str], EmbedChunkPlan] = {}
    for service in services:
        key = (
            service.model_id.casefold(),
            service.quant.casefold(),
            service.backend.casefold(),
        )
        newest = max(
            (
                record
                for record in records.values()
                if record.harness == RETRIEVAL_HARNESS_VERSION
                and record.digest == retrieval_digest()
                and record.model_id.casefold() == key[0]
                and record.quant.casefold() == key[1]
                and record.backend.casefold() == key[2]
                and record.control_passed
                and record.pool_recovers is True
                and record.chunk is not None
            ),
            key=lambda record: record.at,
            default=None,
        )
        if newest is not None and newest.chunk is not None:
            plans[key] = EmbedChunkPlan(
                newest.chunk.chunk_words,
                newest.chunk.chunk_tokens,
            )
    return plans


def _embedding_autochunk_enabled() -> bool:
    return os.environ.get("NMESH_EMBED_AUTOCHUNK", "").casefold() in {
        "1", "true", "yes",
    }


def _embedding_chunks(
    value: str,
    chunk_words: int,
    chunk_tokens: int,
) -> list[str]:
    words = value.split()
    if len(words) > chunk_words:
        return [
            " ".join(words[index:index + chunk_words])
            for index in range(0, len(words), chunk_words)
        ]
    return [
        value[index:index + chunk_tokens]
        for index in range(0, len(value), chunk_tokens)
    ]


def _base_url(service: PlannedService) -> str:
    return ollama_base_url() if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )


def _delegation_identity(service: PlannedService) -> RoleIdentity:
    return RoleIdentity(
        model_id=service.model_id,
        quant=service.quant,
        backend=service.backend,
        artifact=service_fingerprint(service.backend, service.model_ref) or "",
    )


def _delegation_pair(
    plan: Plan,
) -> tuple[PlannedService | None, PlannedService | None, str]:
    lead_name = plan.routing.role_to_service.get("chat", "")
    lead = next((item for item in plan.services if item.name == lead_name), None)
    if lead is None:
        return None, None, "no_worker"
    generative = [
        item for item in plan.services
        if item.name != lead.name
        and set(item.roles) & {"chat", "code", "worker"}
    ]
    if not generative:
        return lead, None, "no_worker"
    worker_name = plan.routing.role_to_service.get("worker", "")
    worker = next(
        (
            item for item in generative
            if item.name == worker_name
        ),
        None,
    )
    if worker is None:
        worker = min(
            generative,
            key=lambda item: (item.memory.weight_bytes, item.model_id),
        )
    if lead.name in plan.swap_group or worker.name in plan.swap_group:
        return lead, worker, "not_coresident"
    return lead, worker, ""


def _delegation_gate(
    plan: Plan,
) -> tuple[
    PlannedService | None,
    PlannedService | None,
    DelegationRecord | None,
    str,
    str,
]:
    lead, worker, pair_reason = _delegation_pair(plan)
    if pair_reason or lead is None or worker is None:
        return lead, worker, None, pair_reason, pair_reason
    record = best_for(
        load_cache(),
        _delegation_identity(lead),
        _delegation_identity(worker),
        PROTOCOL_VERSION,
    )
    decision, reason = decide(record)
    return lead, worker, record, decision, reason


def _delegation_gate_error(
    reason: str, record: DelegationRecord | None,
    lead: PlannedService | None = None,
    worker: PlannedService | None = None,
) -> str:
    if reason == "not_coresident" and lead is not None and worker is not None:
        return i18n.t(
            "err.delegate_not_coresident",
            i18n.lang(),
            lead=lead.name,
            worker=worker.name,
        )
    if record is None:
        return i18n.t("err.delegate_gate", i18n.lang(), reason=reason)
    return i18n.t(
        "err.delegate_gate_stats",
        i18n.lang(),
        reason=reason,
        delegated=record.delegated_passed,
        lead=record.lead_passed,
        p=record.delegated_p,
    )


def _upstream_body(request: Mapping[str, object], service: PlannedService) -> dict[str, object]:
    body = dict(request)
    body["model"] = service.model_ref
    if service.backend == "ollama":
        body["keep_alive"] = "5m" if service.resident else "30s"
    return body


def _reserved_tokens(request: Mapping[str, object]) -> int:
    values = [0]
    for key in ("max_tokens", "max_completion_tokens"):
        value = request.get(key)
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values)


def _created_timestamp(plan: Plan) -> int:
    try:
        return int(float(plan.created_at))
    except (TypeError, ValueError):
        try:
            created = datetime.fromisoformat(plan.created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return int(created.timestamp())
        except (TypeError, ValueError):
            return int(time.time())


def _chat_service(plan: Plan) -> PlannedService | None:
    name = plan.routing.role_to_service.get("chat", "")
    return next((item for item in plan.services if item.name == name), None)


def _service_is_running_llamacpp(service: PlannedService) -> bool:
    if service.backend != "llamacpp":
        return False
    try:
        runtime = runtime_status()
    except (OSError, ValueError, RuntimeError):
        return False
    return any(
        item.get("service") == service.name
        and bool(item.get("running", False))
        and item.get("backend", service.backend) == "llamacpp"
        for item in runtime.services
    )


async def _routing_token_hint(
    request: Mapping[str, object], plan: Plan
) -> int | None:
    chat = _chat_service(plan)
    if chat is None:
        return None
    content = _content(request)
    count = estimate_tokens(
        content,
        calibration_for(calibration_key(chat.model_id, _is_chat_request(request))),
    )
    threshold = chat.context * 0.8 - _reserved_tokens(request)
    if (
        threshold > 0
        and 0.5 * threshold <= count <= 2 * threshold
        and _service_is_running_llamacpp(chat)
    ):
        assert httpx is not None
        client = httpx.AsyncClient()
        try:
            exact = await exact_tokens(_base_url(chat), content, client)
            if exact is not None:
                count = exact
        finally:
            await client.aclose()
    return count


async def _record_prompt_calibration(
    service: PlannedService,
    request: Mapping[str, object],
    usage: object,
) -> None:
    if "tools" in request or "functions" in request:
        return
    if not isinstance(usage, Mapping):
        return
    value = usage.get("prompt_tokens")
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return
    try:
        prompt_tokens = int(value)
    except (TypeError, ValueError):
        return
    if prompt_tokens < 0:
        return
    try:
        await asyncio.to_thread(
            record_token_calibration,
            calibration_key(service.model_id, _is_chat_request(request)),
            _content(request),
            prompt_tokens,
        )
    except Exception:  # noqa: BLE001, S110
        pass


def _completion_not_supported(
    path: str, status_code: int, backend: str
) -> HTTPException | None:
    if status_code != 404 or path != "/v1/completions":
        return None
    return HTTPException(
        status_code=502,
        detail=f"Backend {backend} does not support /v1/completions",
    )


def _embedding_input(request: Mapping[str, object]) -> tuple[str | None, bool]:
    value = request.get("input")
    if isinstance(value, str):
        return value, False
    if not isinstance(value, list):
        return None, False
    if len(value) > 1:
        return None, True
    if len(value) == 1 and isinstance(value[0], str):
        return value[0], False
    return None, False


def _split_embedding_input(value: str) -> tuple[str, str] | None:
    words = value.split()
    if len(words) < 2:
        return None
    midpoint = len(words) // 2
    head = " ".join(words[:midpoint])
    tail = " ".join(words[midpoint:])
    return head, tail


async def _confirm_embedding_truncation(
    body: Mapping[str, object],
    url: str,
    input_text: str,
    cap: int,
) -> int | None:
    split = _split_embedding_input(input_text)
    if split is None:
        return None
    probe = {
        "model": body.get("model"),
        "input": list(split),
    }
    assert httpx is not None
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=CONNECT_TIMEOUT)
        ) as client:
            response = await client.post(url, json=probe)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    tokens = (
        _upstream_int(usage.get("prompt_tokens"))
        if isinstance(usage, dict) else None
    )
    return tokens if tokens is not None and tokens > cap + 2 else None


def _embedding_truncation_error(cap: int, lower_bound: int) -> Response:
    payload = {
        "error": {
            "message": (
                f"Embedding input exceeds the measured served cap of {cap} "
                f"tokens; confirmation measured at least {lower_bound} tokens."
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    return Response(
        content=json.dumps(payload),
        status_code=400,
        media_type="application/json",
    )


def _metric_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"non-numeric metric value: {value!r}")
    return float(value)


def _prometheus_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prometheus_labels(labels: Mapping[str, object]) -> str:
    if not labels:
        return ""
    values = ", ".join(
        f'{name}="{_prometheus_escape(value)}"'
        for name, value in sorted(labels.items())
    )
    return "{" + values + "}"


def _calibration_metrics(services: list[PlannedService]) -> dict[str, dict[str, object]]:
    sums = all_sums()
    metrics: dict[str, dict[str, object]] = {}
    for service in services:
        for kind, chat in (("chat", True), ("text", False)):
            calibration = fit(
                sums.get(calibration_key(service.model_id, chat), Sums())
            )
            metrics[f"{service.name}|{kind}"] = {
                "model": service.model_id,
                "kind": kind,
                "cjk_per_char": calibration.cjk_per_char,
                "other_per_char": calibration.other_per_char,
                "samples": calibration.samples,
                "measured": calibration.measured,
            }
    return metrics


def _prometheus_text(
    limiter: SlotLimiter, services: list[PlannedService] | None = None
) -> str:
    families: dict[str, tuple[str, str, list[tuple[Mapping[str, object], object]]]] = {
        "nmesh_telemetry_samples": (
            "Number of telemetry samples.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_decode_tokens_per_second_median": (
            "Median measured or approximate decode throughput in tokens per second.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_prefill_tokens_per_second_median": (
            "Median measured prefill throughput in tokens per second.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_ttft_seconds_median": (
            "Median measured or approximate time to first token in seconds.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_ttft_seconds_p95": (
            "95th percentile measured or approximate time to first token in seconds.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_total_seconds_median": (
            "Median measured or approximate total request time in seconds.",
            "gauge",
            [],
        ),
        "nmesh_concurrency_limit": (
            "Configured backend concurrency slot limit.",
            "gauge",
            [],
        ),
        "nmesh_concurrency_in_flight": (
            "Backend requests currently occupying concurrency slots.",
            "gauge",
            [],
        ),
        "nmesh_concurrency_waiting": (
            "Backend requests waiting for concurrency slots.",
            "gauge",
            [],
        ),
        "nmesh_token_calibration_cjk_per_char": (
            "Calibrated CJK characters per prompt token.",
            "gauge",
            [],
        ),
        "nmesh_token_calibration_other_per_char": (
            "Calibrated non-CJK characters per prompt token.",
            "gauge",
            [],
        ),
        "nmesh_token_calibration_samples": (
            "Number of exact prompt-token calibration samples.",
            "gauge",
            [],
        ),
    }
    for service, groups in summary_by_approximate().items():
        for approximate, metrics in groups.items():
            labels = {"approximate": str(approximate).lower(), "service": service}
            families["nmesh_telemetry_samples"][2].append((labels, metrics["samples"]))
            metric_names = {
                "decode_tps_median": "nmesh_telemetry_decode_tokens_per_second_median",
                "prefill_tps_median": "nmesh_telemetry_prefill_tokens_per_second_median",
                "ttft_s_median": "nmesh_telemetry_ttft_seconds_median",
                "ttft_s_p95": "nmesh_telemetry_ttft_seconds_p95",
                "total_s_median": "nmesh_telemetry_total_seconds_median",
            }
            for key, metric_name in metric_names.items():
                if key in metrics:
                    families[metric_name][2].append((labels, metrics[key]))
    for service, counts in limiter.metrics().items():
        labels = {"service": service}
        families["nmesh_concurrency_limit"][2].append((labels, counts["limit"]))
        families["nmesh_concurrency_in_flight"][2].append((labels, counts["in_flight"]))
        families["nmesh_concurrency_waiting"][2].append((labels, counts["waiting"]))
    for service_kind, calibration in _calibration_metrics(services or []).items():
        labels = {
            "kind": str(calibration["kind"]),
            "measured": str(calibration["measured"]).lower(),
            "model": str(calibration["model"]),
            "service": service_kind.rsplit("|", 1)[0],
        }
        families["nmesh_token_calibration_cjk_per_char"][2].append(
            (labels, calibration["cjk_per_char"])
        )
        families["nmesh_token_calibration_other_per_char"][2].append(
            (labels, calibration["other_per_char"])
        )
        families["nmesh_token_calibration_samples"][2].append(
            (labels, calibration["samples"])
        )
    lines: list[str] = []
    for name, (help_text, metric_type, values) in families.items():
        lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"))
        lines.extend(
            f"{name}{_prometheus_labels(labels)} {_metric_number(value):g}"
            for labels, value in values
        )
    return "\n".join(lines) + "\n"


class _PlanState:
    def __init__(
        self, plan: Plan, explicit: bool, gate: SwapGate, limiter: SlotLimiter
    ) -> None:
        self.plan = plan
        self.telemetry_keys: dict[str, str] = {}
        self.mtime: float | None = None
        self._mtime_ns: int | None = None
        self.embed_input_caps: dict[tuple[str, str, str], int] = {}
        self.embed_chunk_plans: dict[tuple[str, str, str], EmbedChunkPlan] = {}
        self._explicit = explicit
        self._gate = gate
        self._limiter = limiter
        self.reload(plan)
        if not explicit:
            stamp = self._plan_stamp()
            if stamp is not None:
                self.mtime, self._mtime_ns = stamp

    def snapshot(self) -> tuple[Plan, dict[str, str]]:
        return self.plan, self.telemetry_keys.copy()

    def reload(self, plan: Plan) -> None:
        gpu = plan.profile.gpus[0].name if plan.profile.gpus else "cpu"
        self.plan = plan
        self.embed_input_caps = _embed_input_caps(plan)
        self.embed_chunk_plans = _embed_chunk_plans(plan)
        self.telemetry_keys = {
            service.name: benchmark_key(
                service.model_id, service.quant, service.backend, gpu,
                service.n_gpu_layers, service.kv_quant, service.spec,
                service.n_cpu_moe, service.tensor_split,
            )
            for service in plan.services
        }
        self._limiter.size(plan)
        self._gate.invalidate()

    def _plan_stamp(self) -> tuple[float, int] | None:
        try:
            stat = PLAN_PATH.stat()
            return stat.st_mtime, stat.st_mtime_ns
        except OSError:
            return None

    def _reload_from_disk(self, force: bool = False) -> bool | None:
        stamp = self._plan_stamp()
        if stamp is None:
            return None if force else False
        mtime, mtime_ns = stamp
        if not force and mtime == self.mtime and mtime_ns == self._mtime_ns:
            return False
        try:
            loaded = load_plan(PLAN_PATH)
        except Exception:  # noqa: BLE001
            return None if force else False
        if loaded is None:
            return None if force else False
        changed = loaded != self.plan
        self.mtime = mtime
        self._mtime_ns = mtime_ns
        if changed:
            self.reload(loaded)
        return changed

    def maybe_reload(self) -> bool:
        if self._explicit:
            return False
        return bool(self._reload_from_disk())

    def force_reload(self) -> bool | None:
        return self._reload_from_disk(force=True)


async def _slot_progress(
    services: Sequence[PlannedService], entries: list[Job]
) -> dict[str, dict[str, int]]:
    """Decode progress for running jobs, from llama.cpp's /slots endpoint.

    A job is annotated only when the mapping is unambiguous: its service has
    exactly one running job here and the backend reports exactly one
    processing slot. Every failure mode (non-llamacpp backend, /slots
    disabled, timeout, count drift) leaves the field absent rather than
    reporting a wrong value.
    """
    running: dict[str, list[Job]] = {}
    for job in entries:
        if job.state == "running":
            running.setdefault(job.service, []).append(job)
    if not running or httpx is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for service_name, service_jobs in running.items():
        if len(service_jobs) != 1:
            continue
        service = next(
            (item for item in services if item.name == service_name), None
        )
        if service is None or not _service_is_running_llamacpp(service):
            continue
        try:
            async with httpx.AsyncClient(
                base_url=_base_url(service), timeout=0.8
            ) as client:
                response = await client.get("/slots")
            if response.status_code != 200:
                continue
            slots = response.json()
        except (httpx.HTTPError, ValueError):
            continue
        if not isinstance(slots, list):
            continue
        processing = [
            slot for slot in slots
            if isinstance(slot, dict) and slot.get("is_processing") is True
        ]
        if len(processing) != 1:
            continue
        next_token = processing[0].get("next_token")
        # /slots reports next_token as a one-element list.
        if isinstance(next_token, list) and next_token:
            token_info = next_token[0]
        else:
            token_info = next_token
        progress: dict[str, int] = {}
        if isinstance(token_info, dict):
            decoded = _upstream_int(token_info.get("n_decoded"))
            if decoded is not None:
                progress["decoded"] = decoded
            remaining = _upstream_int(token_info.get("n_remain"))
            if remaining is not None:
                progress["remaining"] = remaining
        if progress:
            out[service_jobs[0].id] = progress
    return out


def create_app(
    plan: Plan | None = None,
    watchdog: bool = False,
    watchdog_interval: float = 15.0,
) -> FastAPIApp:
    if FastAPI is None:
        raise ImportError("Install nmesh[gateway] to use the gateway")
    explicit = plan is not None
    selected = plan if explicit else load_plan(PLAN_PATH)
    if selected is None:
        raise FileNotFoundError("No plan found")
    last_use = _LastUse(selected.services)

    @asynccontextmanager
    async def lifespan(_app: object):
        tasks: list[asyncio.Task[None]] = []
        if watchdog:
            async def watch() -> None:
                while True:
                    await asyncio.sleep(watchdog_interval)
                    try:
                        await asyncio.to_thread(heartbeat)
                    except Exception:  # noqa: BLE001, S110
                        pass

            tasks.append(asyncio.create_task(watch()))
        if KEEP_ALIVE > 0:
            async def reap_loop() -> None:
                interval = max(1.0, min(15.0, KEEP_ALIVE / 2))
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await _reap()
                    except Exception:  # noqa: BLE001, S110
                        pass

            tasks.append(asyncio.create_task(reap_loop()))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="nmesh gateway", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, error: HTTPException
    ) -> Response:
        status_code = error.status_code
        error_type = (
            "invalid_request_error" if status_code < 500 else "server_error"
        )
        payload = {
            "error": {
                "message": str(error.detail),
                "type": error_type,
                "code": status_code,
            }
        }
        return Response(
            content=json.dumps(payload),
            status_code=status_code,
            media_type="application/json",
            headers=dict(error.headers or {}),
        )

    gate = SwapGate()
    limiter = SlotLimiter()
    jobs = JobRegistry()
    in_flight = _InFlight()
    plan_state = _PlanState(selected, explicit, gate, limiter)
    revive_locks: dict[str, asyncio.Lock] = {}

    async def _reap() -> None:
        plan_state.maybe_reload()
        current, _ = plan_state.snapshot()
        last_use.ensure(current.services)
        runtime = await asyncio.to_thread(runtime_status)
        now = time.monotonic()
        for item in runtime.services:
            name = item.get("service")
            if not isinstance(name, str) or not item.get("running"):
                continue
            if in_flight.count(name) or last_use.age(name, now) is None:
                continue
            idle_for = last_use.age(name, now)
            if (
                idle_for is not None
                and idle_for > KEEP_ALIVE
                and await asyncio.to_thread(unload, name)
            ):
                gate.invalidate()

    app.state.reap = _reap
    app.state.in_flight = in_flight
    api_key = os.environ.get("NMESH_API_KEY")
    api_key_bytes = api_key.encode("utf-8") if api_key is not None else None

    @app.middleware("http")
    async def authenticate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        origin = request.headers.get("origin")
        # Browser frontends (Open WebUI-style) preflight with OPTIONS before
        # sending Authorization; answer CORS before the key check, which would
        # otherwise reject every preflight. The gateway binds 127.0.0.1 and the
        # bearer key still guards real requests, so reflecting the origin is safe.
        if (
            origin
            and path.startswith("/v1/")
            and request.method == "OPTIONS"
        ):
            requested_headers = request.headers.get(
                "access-control-request-headers", "authorization,content-type"
            )
            return Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": requested_headers,
                    "Access-Control-Max-Age": "600",
                    "Vary": "Origin",
                },
            )
        if api_key_bytes is not None and path.startswith(
            ("/v1/", "/metrics", "/admin/", "/logs")
        ):
            authorization = request.headers.get("authorization", "")
            prefix = "Bearer "
            presented = authorization[len(prefix):] if authorization.startswith(prefix) else ""
            if not secrets.compare_digest(presented.encode("latin-1"), api_key_bytes):
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": "Invalid or missing API key",
                            "type": "invalid_request_error",
                            "code": 401,
                        }
                    }),
                    status_code=401,
                    media_type="application/json",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        response = await call_next(request)
        if origin and path.startswith("/v1/"):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        return response

    async def proxy(request: dict[str, object], service: PlannedService,
                    plan_snapshot: Plan, telemetry_keys: Mapping[str, str],
                    path: str, instrument: bool = True,
                    limit_slots: bool = False) -> object:
        started = time.perf_counter()
        last_use.touch(service.name)
        slot_token: object | None = None
        job = jobs.submit(service.name, path)
        if limit_slots:
            deadline = time.monotonic() + QUEUE_TIMEOUT
            while job.state == "queued" and slot_token is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                slot_token = await limiter.acquire(
                    service, min(0.5, remaining)
                )
            if job.state == "cancelled":
                if slot_token is not None:
                    limiter.release(slot_token)
                    slot_token = None
                raise HTTPException(
                    status_code=409,
                    detail=f"job {job.id} cancelled while queued",
                )
            if slot_token is None:
                position = jobs.position(job)
                jobs.finish(job, ok=False, detail="queue_timeout")
                slots = max(1, service.memory.parallel_slots)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Service {service.name} is at its concurrency limit "
                        f"({slots} slots); job {job.id} timed out at "
                        f"queue position {position}"
                    ),
                    headers={"Retry-After": "1"},
                )
        if not jobs.start(job):
            if slot_token is not None:
                limiter.release(slot_token)
            raise HTTPException(
                status_code=409,
                detail=f"job {job.id} cancelled while queued",
            )
        locked = service.name in plan_snapshot.swap_group
        if locked:

            def _ensure_target() -> None:
                ensure_running(service.name, plan_snapshot)

            try:
                await asyncio.wait_for(
                    gate.acquire(service.name, _ensure_target),
                    timeout=300.0,
                )
            except asyncio.TimeoutError as error:
                raise HTTPException(status_code=504, detail="Timed out waiting for service swap") from error
            except Exception as error:
                jobs.finish(job, ok=False, detail="ensure_failed")
                raise HTTPException(status_code=503, detail=str(error)) from error
        if not locked and service.name in await asyncio.to_thread(idle_services):
            revive_lock = revive_locks.setdefault(service.name, asyncio.Lock())
            async with revive_lock:
                if service.name in await asyncio.to_thread(idle_services):
                    try:
                        await asyncio.to_thread(ensure_running, service.name, plan_snapshot)
                    except Exception as error:
                        jobs.finish(job, ok=False, detail="ensure_failed")
                        raise HTTPException(status_code=503, detail=str(error)) from error
        body = _upstream_body(request, service)
        url = f"{_base_url(service)}{path}"
        embedding_cap = (
            plan_state.embed_input_caps.get((
                service.model_id.casefold(),
                service.quant.casefold(),
                service.backend.casefold(),
            ))
            if path == "/v1/embeddings" and service.roles == ["embed"]
            else None
        )
        assert httpx is not None
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=CONNECT_TIMEOUT))
        ticket = in_flight.enter(service.name)
        if request.get("stream"):
            stream_options = request.get("stream_options")
            request_wants_usage = (
                isinstance(stream_options, Mapping)
                and bool(stream_options.get("include_usage"))
            )
            if service.backend == "llamacpp" and not path.startswith("/v1/messages"):
                upstream_stream_options = (
                    dict(stream_options) if isinstance(stream_options, Mapping) else {}
                )
                upstream_stream_options["include_usage"] = True
                body["stream_options"] = upstream_stream_options
            try:
                upstream_request = client.build_request("POST", url, json=body)
                try:
                    upstream = await client.send(upstream_request, stream=True)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    try:
                        await asyncio.to_thread(ensure_running, service.name, plan_snapshot)
                    except Exception as error:
                        raise HTTPException(status_code=502, detail=str(error)) from error
                    upstream_request = client.build_request(
                        "POST", f"{_base_url(service)}{path}", json=body
                    )
                    upstream = await client.send(upstream_request, stream=True)
            except HTTPException:
                await client.aclose()
                in_flight.leave(service.name, ticket)
                last_use.touch(service.name)
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                if job is not None:
                    jobs.finish(job, ok=False, detail="upstream_unreachable")
                raise
            except httpx.HTTPError as error:
                await client.aclose()
                in_flight.leave(service.name, ticket)
                last_use.touch(service.name)
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                if job is not None:
                    jobs.finish(job, ok=False, detail=str(error))
                raise HTTPException(status_code=502, detail=str(error)) from error
            if upstream.status_code >= 400:
                content = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                in_flight.leave(service.name, ticket)
                last_use.touch(service.name)
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                unsupported = _completion_not_supported(
                    path, upstream.status_code, service.backend
                )
                if unsupported is not None:
                    if job is not None:
                        jobs.finish(job, ok=False,
                                    detail=f"upstream_{upstream.status_code}")
                    raise unsupported
                if job is not None:
                    jobs.finish(job, ok=False,
                                detail=f"upstream_{upstream.status_code}")
                return Response(content=content, status_code=upstream.status_code,
                                media_type=upstream.headers.get("content-type"))

            stream_completed = False

            async def stream() -> AsyncIterator[bytes]:
                nonlocal stream_completed
                buffer = b""
                tokens = 0
                usage: dict[str, object] | None = None
                timings: dict[str, object] | None = None
                first_line_time: float | None = None
                last_line_time: float | None = None

                def transform(line: bytes) -> bytes | None:
                    nonlocal first_line_time, last_line_time, tokens, usage, timings
                    ending = b""
                    content = line
                    if content.endswith(b"\n"):
                        content = content[:-1]
                        ending = b"\n"
                        if content.endswith(b"\r"):
                            content = content[:-1]
                            ending = b"\r\n"
                    if not content.startswith(b"data: ") or content == b"data: [DONE]":
                        return line
                    try:
                        payload = json.loads(content[6:])
                    except json.JSONDecodeError:
                        payload = None
                    if not isinstance(payload, dict):
                        now = time.perf_counter()
                        tokens += 1
                        first_line_time = first_line_time or now
                        last_line_time = now
                        return line
                    candidate_usage = payload.get("usage")
                    candidate_timings = payload.get("timings")
                    usage_only = (
                        isinstance(candidate_usage, dict)
                        and isinstance(payload.get("choices"), list)
                        and not payload["choices"]
                    )
                    if isinstance(candidate_usage, dict):
                        usage = candidate_usage
                    if isinstance(candidate_timings, dict):
                        timings = candidate_timings
                    if usage_only and not request_wants_usage:
                        return None
                    timing_only = (
                        isinstance(candidate_timings, dict)
                        and (
                            not isinstance(payload.get("choices"), list)
                            or not payload["choices"]
                        )
                    )
                    if not usage_only and not timing_only:
                        now = time.perf_counter()
                        tokens += 1
                        first_line_time = first_line_time or now
                        last_line_time = now
                    nested_message = payload.get("message")
                    if "model" in payload or (
                        isinstance(nested_message, dict) and "model" in nested_message
                    ):
                        client_model = request.get("model", service.model_id)
                        if "model" in payload:
                            payload["model"] = client_model
                        if isinstance(nested_message, dict) and "model" in nested_message:
                            nested_message["model"] = client_model
                        content = (
                            b"data: "
                            + json.dumps(payload, separators=(",", ":")).encode()
                        )
                    return content + ending

                try:
                    async for chunk in upstream.aiter_bytes():
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            output = transform(line + b"\n")
                            if output is not None:
                                yield output
                    if buffer:
                        output = transform(buffer)
                        if output is not None:
                            yield output
                    stream_completed = True
                except httpx.HTTPError as error:
                    raise HTTPException(status_code=502, detail=str(error)) from error
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    in_flight_peak = in_flight.leave(service.name, ticket)
                    last_use.touch(service.name)
                    if locked:
                        gate.release()
                    if limit_slots:
                        limiter.release(slot_token)
                    if job is not None:
                        jobs.finish(
                            job,
                            ok=stream_completed,
                            detail=None if stream_completed else "stream_interrupted",
                        )
                    if instrument:
                        span = (last_line_time - first_line_time
                                if first_line_time is not None and last_line_time is not None
                                else 0.0)
                        timing_decode, timing_prefill = _timing_metrics(timings)
                        usage_completion = (
                            _upstream_int(usage.get("completion_tokens"))
                            if usage is not None else None
                        )
                        exact = (
                            usage is not None
                            and usage_completion is not None
                            and "completion_tokens" in usage
                            and (service.backend == "llamacpp" or request_wants_usage)
                        )
                        completion_tokens = (
                            usage_completion if exact and usage_completion is not None
                            else tokens
                        )
                        decode = (
                            timing_decode
                            if timing_decode is not None else (
                                max(completion_tokens - 1, 0) / span
                                if completion_tokens >= 16 and span > 0 else None
                            )
                        )
                        if timing_decode is not None:
                            exact = True
                        try:
                            await asyncio.to_thread(record_telemetry, Sample(
                                service.name, telemetry_keys[service.name], decode,
                                first_line_time - started if first_line_time is not None else None,
                                time.perf_counter() - started, completion_tokens, time.time(),
                                not exact,
                                timing_prefill,
                                in_flight_peak,
                                prompt_tokens=_prompt_depth(usage, timings),
                            ))
                        except Exception:  # noqa: BLE001, S110
                            pass
                    await _record_prompt_calibration(service, request, usage)
            return StreamingResponse(
                stream(), media_type="text/event-stream",
                headers=(
                    {"X-Nmesh-Job-Id": job.id} if job is not None else {}
                ),
            )
        try:
            async def post_upstream(payload: Mapping[str, object]) -> httpx.Response:
                try:
                    return await client.post(url, json=payload)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    try:
                        await asyncio.to_thread(
                            ensure_running, service.name, plan_snapshot
                        )
                    except Exception as error:
                        raise HTTPException(status_code=502, detail=str(error)) from error
                    return await client.post(
                        f"{_base_url(service)}{path}", json=payload
                    )

            chunk_plan = (
                plan_state.embed_chunk_plans.get((
                    service.model_id.casefold(),
                    service.quant.casefold(),
                    service.backend.casefold(),
                ))
                if path == "/v1/embeddings" and service.roles == ["embed"]
                and _embedding_autochunk_enabled()
                else None
            )
            autochunk_data: dict[str, object] | None = None
            autochunk_used = False
            autochunk_piece_count: int | None = None
            autochunk_chunk_words: int | None = None
            autochunk_chunk_tokens: int | None = None
            input_value = request.get("input")
            input_values: list[str] | None = None
            if isinstance(input_value, str):
                input_values = [input_value]
            elif (
                isinstance(input_value, list)
                and all(isinstance(item, str) for item in input_value)
            ):
                input_values = list(input_value)
            if chunk_plan is not None and input_values:
                piece_groups = [
                    (
                        _embedding_chunks(
                            item,
                            chunk_plan.chunk_words,
                            chunk_plan.chunk_tokens,
                        )
                        if (
                            len(item.split()) > chunk_plan.chunk_words
                            or len(item) > chunk_plan.chunk_tokens
                        )
                        else [item]
                    )
                    for item in input_values
                ]
                if any(len(group) > 1 for group in piece_groups):
                    outputs: list[list[float]] = []
                    total_tokens = 0
                    total_pieces = 0
                    autochunk_data = {
                        "object": "list",
                        "model": request.get("model", service.model_id),
                        "data": [],
                        "usage": {},
                    }
                    try:
                        for pieces in piece_groups:
                            piece_body = dict(body)
                            piece_body["input"] = pieces[0] if len(pieces) == 1 else pieces
                            piece_response = await post_upstream(piece_body)
                            if piece_response.status_code >= 400:
                                autochunk_data = None
                                break
                            piece_payload = piece_response.json()
                            if not isinstance(piece_payload, Mapping):
                                autochunk_data = None
                                break
                            piece_data = piece_payload.get("data")
                            usage = piece_payload.get("usage")
                            prompt_tokens = (
                                _upstream_int(usage.get("prompt_tokens"))
                                if isinstance(usage, Mapping) else None
                            )
                            if (
                                not isinstance(piece_data, list)
                                or len(piece_data) != len(pieces)
                                or prompt_tokens is None
                                or prompt_tokens < 0
                            ):
                                autochunk_data = None
                                break
                            if not all(isinstance(item, Mapping) for item in piece_data):
                                autochunk_data = None
                                break
                            vectors = [item.get("embedding") for item in piece_data]
                            if not all(
                                isinstance(vector, list)
                                and all(
                                    isinstance(value, (int, float))
                                    and not isinstance(value, bool)
                                    for value in vector
                                )
                                for vector in vectors
                            ):
                                autochunk_data = None
                                break
                            outputs.append(
                                vectors[0]
                                if len(pieces) == 1
                                else pool_embeddings(vectors)
                            )
                            total_tokens += prompt_tokens
                            total_pieces += len(pieces)
                    except (httpx.HTTPError, TypeError, ValueError, KeyError):
                        autochunk_data = None
                    if autochunk_data is not None:
                        autochunk_data["data"] = [
                            {
                                "object": "embedding",
                                "index": index,
                                "embedding": vector,
                            }
                            for index, vector in enumerate(outputs)
                        ]
                        autochunk_data["usage"] = {
                            "prompt_tokens": total_tokens,
                            "total_tokens": total_tokens,
                        }
                        autochunk_piece_count = total_pieces
                        autochunk_chunk_words = chunk_plan.chunk_words
                        autochunk_chunk_tokens = chunk_plan.chunk_tokens
                        autochunk_used = True
            if autochunk_data is not None:
                data = autochunk_data
            else:
                response = await post_upstream(body)
                content = response.content
                if response.status_code >= 400:
                    unsupported = _completion_not_supported(
                        path, response.status_code, service.backend
                    )
                    if unsupported is not None:
                        if job is not None:
                            jobs.finish(
                                job, ok=False,
                                detail=f"upstream_{response.status_code}",
                            )
                        raise unsupported
                    if job is not None:
                        jobs.finish(
                            job, ok=False,
                            detail=f"upstream_{response.status_code}",
                        )
                    return Response(content=content, status_code=response.status_code,
                                    media_type=response.headers.get("content-type"))
                data = json.loads(content)
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            if job is not None:
                jobs.finish(job, ok=False, detail=str(error))
            raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            await client.aclose()
            in_flight_peak = in_flight.leave(service.name, ticket)
            last_use.touch(service.name)
            if locked:
                gate.release()
            if limit_slots:
                limiter.release(slot_token)
            if job is not None and sys.exc_info()[1] is not None:
                jobs.finish(job, ok=False, detail=str(sys.exc_info()[1]))
        if isinstance(data, dict) and "model" in data:
            data["model"] = request.get("model", service.model_id)
        embedding_headers: dict[str, str] = {}
        embedding_chunk_headers: dict[str, str] = {}
        if (
            autochunk_used
            and autochunk_piece_count is not None
            and autochunk_chunk_words is not None
            and autochunk_chunk_tokens is not None
        ):
            embedding_chunk_headers = {
                "X-Nmesh-Embedding-Chunked": str(autochunk_piece_count),
                "X-Nmesh-Embedding-Chunk-Words": str(autochunk_chunk_words),
                "X-Nmesh-Embedding-Chunk-Chars": str(autochunk_chunk_tokens),
            }
        skip_embedding_guard = autochunk_used
        if embedding_cap is not None and isinstance(data, dict) and not skip_embedding_guard:
            usage = data.get("usage")
            prompt_tokens = (
                _upstream_int(usage.get("prompt_tokens"))
                if isinstance(usage, dict) else None
            )
            if prompt_tokens is not None and prompt_tokens >= embedding_cap:
                input_text, multi_input = _embedding_input(request)
                if multi_input:
                    embedding_headers["X-Nmesh-Embedding-Truncation"] = "unverified"
                elif input_text is not None:
                    lower_bound = await _confirm_embedding_truncation(
                        body, url, input_text, embedding_cap
                    )
                    if lower_bound is not None:
                        if job is not None:
                            jobs.finish(
                                job, ok=False,
                                detail=f"truncation_guard_{embedding_cap}",
                            )
                        return _embedding_truncation_error(
                            embedding_cap, lower_bound
                        )
        if instrument:
            usage = data.get("usage") if isinstance(data, dict) else None
            timings = data.get("timings") if isinstance(data, dict) else None
            timing_decode, timing_prefill = _timing_metrics(timings)
            usage_completion = (
                _upstream_int(usage.get("completion_tokens"))
                if isinstance(usage, dict) else None
            )
            completion_tokens = (
                usage_completion if usage_completion is not None else 0
            )
            exact = (
                isinstance(usage, dict)
                and usage_completion is not None
                and "completion_tokens" in usage
            )
            if timing_decode is not None:
                exact = True
            try:
                await asyncio.to_thread(record_telemetry, Sample(
                    service.name, telemetry_keys[service.name], timing_decode, None,
                    time.perf_counter() - started, completion_tokens, time.time(),
                    not exact,
                    timing_prefill,
                    in_flight_peak,
                    prompt_tokens=_prompt_depth(usage, timings),
                ))
            except Exception:  # noqa: BLE001, S110
                pass
        await _record_prompt_calibration(
            service,
            request,
            data.get("usage") if isinstance(data, dict) else None,
        )
        embedding_headers.update(embedding_chunk_headers)
        if embedding_headers:
            if job is not None:
                embedding_headers["X-Nmesh-Job-Id"] = job.id
                jobs.finish(job, ok=True)
            return Response(
                content=json.dumps(data),
                media_type="application/json",
                headers=embedding_headers,
            )
        if job is not None:
            jobs.finish(job, ok=True)
            return Response(
                content=json.dumps(data),
                media_type="application/json",
                headers={"X-Nmesh-Job-Id": job.id},
            )
        return data

    async def _delegated_completion(
        request: Mapping[str, object], plan_snapshot: Plan
    ) -> dict[str, object]:
        if bool(request.get("stream")):
            raise HTTPException(
                status_code=400,
                detail=i18n.t("err.delegate_stream", i18n.lang()),
            )
        lead, worker, record, decision, reason = _delegation_gate(plan_snapshot)
        if lead is None or worker is None:
            raise HTTPException(
                status_code=409,
                detail=i18n.t("err.delegate_worker", i18n.lang()),
            )
        if decision != "allow":
            raise HTTPException(
                status_code=409,
                detail=_delegation_gate_error(reason, record, lead, worker),
            )
        requested_limit = (
            request.get("max_tokens")
            or request.get("max_completion_tokens")
            or 256
        )
        try:
            max_tokens = (
                int(requested_limit)
                if isinstance(requested_limit, (int, float, str))
                else 256
            )
        except (TypeError, ValueError):
            max_tokens = 256
        prompt = _content(request)
        ledger = Ledger()
        managed: list[tuple[PlannedService, object, _Ticket]] = []

        def concurrency_error(service: PlannedService) -> HTTPException:
            slots = max(1, service.memory.parallel_slots)
            return HTTPException(
                status_code=503,
                detail=(
                    f"Service {service.name} is at its concurrency limit "
                    f"({slots} slots)"
                ),
                headers={"Retry-After": "1"},
            )

        try:
            for service in (lead, worker):
                last_use.touch(service.name)
                if service.name in await asyncio.to_thread(idle_services):
                    revive_lock = revive_locks.setdefault(
                        service.name, asyncio.Lock()
                    )
                    async with revive_lock:
                        if service.name in await asyncio.to_thread(idle_services):
                            try:
                                await asyncio.to_thread(
                                    ensure_running, service.name, plan_snapshot
                                )
                            except Exception as error:
                                raise HTTPException(
                                    status_code=503, detail=str(error)
                                ) from error
                slot = await limiter.acquire(service, QUEUE_TIMEOUT)
                if slot is None:
                    raise concurrency_error(service)
                ticket = in_flight.enter(service.name)
                managed.append((service, slot, ticket))

            def run() -> Delegation:
                assert httpx is not None
                with httpx.Client(
                    timeout=httpx.Timeout(300.0, connect=CONNECT_TIMEOUT)
                ) as client:
                    return delegate(
                        client,
                        prompt,
                        max(1, max_tokens),
                        lead=Endpoint(_base_url(lead), lead.model_ref),
                        worker=Endpoint(_base_url(worker), worker.model_ref),
                        ledger=ledger,
                    )

            try:
                result = await asyncio.to_thread(run)
            except (httpx.HTTPError, ValueError) as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            for service, slot, ticket in reversed(managed):
                in_flight.leave(service.name, ticket)
                limiter.release(slot)
                last_use.touch(service.name)
        role = "worker" if result.accepted else "lead"
        return {
            "id": f"nmesh-delegate-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "nmesh-delegate",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": (
                    ledger.worker.prompt_tokens
                    + ledger.verify.prompt_tokens
                    + ledger.rescue.prompt_tokens
                ),
                "completion_tokens": (
                    ledger.worker.completion_tokens
                    + ledger.verify.completion_tokens
                    + ledger.rescue.completion_tokens
                ),
                "total_tokens": (
                    ledger.worker.prompt_tokens
                    + ledger.verify.prompt_tokens
                    + ledger.rescue.prompt_tokens
                    + ledger.worker.completion_tokens
                    + ledger.verify.completion_tokens
                    + ledger.rescue.completion_tokens
                ),
            },
            "nmesh_role": role,
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status_endpoint() -> dict[str, object]:
        return asdict(runtime_status())

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        ids = ["nmesh-auto"] + [f"nmesh-{service.name}" for service in selected.services]
        _, worker, _, decision, _ = _delegation_gate(selected)
        if worker is not None and decision == "allow":
            ids.append("nmesh-delegate")
        created = _created_timestamp(selected)
        return {"object": "list", "data": [
            {
                "id": item,
                "object": "model",
                "owned_by": "nmesh",
                "created": created,
            }
            for item in ids
        ]}

    @app.get("/v1/models/{model_id}")
    async def model(model_id: str) -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        ids = {"nmesh-auto"} | {
            f"nmesh-{service.name}" for service in selected.services
        }
        _, worker, _, decision, _ = _delegation_gate(selected)
        if worker is not None and decision == "allow":
            ids.add("nmesh-delegate")
        if model_id not in ids:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return {
            "id": model_id,
            "object": "model",
            "owned_by": "nmesh",
            "created": _created_timestamp(selected),
        }

    @app.post("/admin/reload")
    async def reload_endpoint() -> dict[str, object]:
        reloaded = plan_state.force_reload()
        if reloaded is None:
            raise HTTPException(status_code=503, detail="No plan found")
        selected, _ = plan_state.snapshot()
        return {
            "reloaded": reloaded,
            "services": [service.name for service in selected.services],
            "created_at": selected.created_at,
        }

    @app.post("/admin/unload")
    async def unload_all() -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        unloaded: list[str] = []
        runtime = await asyncio.to_thread(runtime_status)
        idle = await asyncio.to_thread(idle_services)
        by_name = {
            item.get("service"): item
            for item in runtime.services
            if isinstance(item.get("service"), str)
        }
        results = [
            await _unload_service(service.name, by_name.get(service.name), idle)
            for service in selected.services
        ]
        if any(result["unloaded"] for result in results):
            gate.invalidate()
        unloaded.extend(
            str(result["service"])
            for result in results
            if result["unloaded"]
        )
        return {"unloaded": unloaded, "results": results}

    @app.post("/admin/unload/{service}")
    async def unload_one(service: str) -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        if service not in {item.name for item in selected.services}:
            raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
        runtime = await asyncio.to_thread(runtime_status)
        idle = await asyncio.to_thread(idle_services)
        item = next(
            (
                entry for entry in runtime.services
                if entry.get("service") == service
            ),
            None,
        )
        result = await _unload_service(service, item, idle)
        if result["unloaded"]:
            gate.invalidate()
        return {
            "unloaded": [service] if result["unloaded"] else [],
            "results": [result],
        }

    @app.get("/admin/running")
    async def running() -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        last_use.ensure(selected.services)
        runtime = await asyncio.to_thread(runtime_status)
        now = time.monotonic()
        by_name = {
            item.get("service"): item
            for item in runtime.services
            if isinstance(item.get("service"), str)
        }
        services = []
        for service in selected.services:
            item = by_name.get(service.name, {})
            idle = bool(item.get("idle", False))
            services.append({
                "service": service.name,
                "running": bool(item.get("running", False)),
                "idle": idle,
                "idle_seconds": last_use.age(service.name, now),
                "in_flight": in_flight.count(service.name),
            })
        return {"keep_alive": KEEP_ALIVE, "services": services}

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        return {
            "services": telemetry_summary(),
            "concurrency": limiter.metrics(),
            "token_calibration": _calibration_metrics(selected.services),
        }

    @app.get("/logs/{service:path}")
    async def logs(service: str, lines: int = 50) -> dict[str, object]:
        try:
            path = log_path(service)
        except ValueError:
            raise HTTPException(
                status_code=404, detail=f"Unknown log: {service}"
            ) from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Unknown log: {service}")
        return {
            "service": service,
            "path": str(path),
            "lines": await asyncio.to_thread(tail_log, service, max(1, min(lines, 1000))),
        }

    @app.get("/metrics/prometheus")
    async def metrics_prometheus() -> object:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        return Response(
            _prometheus_text(limiter, selected.services),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/jobs")
    async def list_jobs(limit: int = 50) -> dict[str, object]:
        entries = jobs.list(limit)
        progress = await _slot_progress(plan_state.snapshot()[0].services, entries)
        return {
            "jobs": [
                {**job.as_dict(), **({"progress": p} if (p := progress.get(job.id)) else {})}
                for job in entries
            ],
            "counts": jobs.counts(),
        }

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, object]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        progress = await _slot_progress(plan_state.snapshot()[0].services, [job])
        data = job.as_dict()
        if job_id in progress:
            data["progress"] = progress[job_id]
        return data

    @app.delete("/v1/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, object]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not jobs.cancel(job):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"job {job.id} is {job.state}; "
                    "only queued jobs can be cancelled"
                ),
            )
        return job.as_dict()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        if request.get("model") == "nmesh-delegate":
            return await _delegated_completion(request, selected)
        token_hint = await _routing_token_hint(request, selected)
        service = _service(selected, route(request, selected, token_hint=token_hint))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/chat/completions",
            limit_slots=True,
        )

    @app.post("/v1/completions")
    async def legacy_completions(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        token_hint = await _routing_token_hint(request, selected)
        service = _service(selected, route(request, selected, token_hint=token_hint))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/completions",
            limit_slots=True,
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        token_hint = await _routing_token_hint(request, selected)
        service = _service(selected, route(request, selected, token_hint=token_hint))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/messages",
            limit_slots=True,
        )

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        token_hint = await _routing_token_hint(request, selected)
        service = _service(selected, route(request, selected, token_hint=token_hint))
        return await proxy(
            request, service, selected, telemetry_keys,
            "/v1/messages/count_tokens", instrument=False,
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        service = _service(selected, selected.routing.role_to_service.get("embed", ""))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/embeddings", instrument=False
        )

    @app.post("/v1/rerank")
    async def rerank(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        # llama.cpp serves rerank OR embeddings per instance (single pooling
        # mode), so rerank needs its own service in the plan.
        name = selected.routing.role_to_service.get("rerank")
        if not name:
            raise HTTPException(
                status_code=501,
                detail=(
                    "this plan has no rerank service; re-run "
                    "`nmesh plan --roles chat,code,embed,rerank`"
                ),
            )
        service = _service(selected, name)
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/rerank", instrument=False
        )

    return app


app = None

__all__ = ["app", "create_app", "route"]
