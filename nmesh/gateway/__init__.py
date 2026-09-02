from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone

from nmesh.bench import benchmark_key
from nmesh.planner import PLAN_PATH, Plan, PlannedService, load_plan
from nmesh.runtime import ensure_running, heartbeat
from nmesh.runtime import status as runtime_status
from nmesh.telemetry import Sample, summary_by_approximate
from nmesh.telemetry import record as record_telemetry
from nmesh.telemetry import summary as telemetry_summary

from .gate import SwapGate
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


def _is_chat_request(request: Mapping[str, object]) -> bool:
    return "messages" in request


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


def _base_url(service: PlannedService) -> str:
    return "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
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
        if value is None:
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
    for service, metrics in limiter.metrics().items():
        labels = {"service": service}
        families["nmesh_concurrency_limit"][2].append((labels, metrics["limit"]))
        families["nmesh_concurrency_in_flight"][2].append((labels, metrics["in_flight"]))
        families["nmesh_concurrency_waiting"][2].append((labels, metrics["waiting"]))
    for service_kind, metrics in _calibration_metrics(services or []).items():
        labels = {
            "kind": metrics["kind"],
            "measured": str(metrics["measured"]).lower(),
            "model": metrics["model"],
            "service": service_kind.rsplit("|", 1)[0],
        }
        families["nmesh_token_calibration_cjk_per_char"][2].append(
            (labels, metrics["cjk_per_char"])
        )
        families["nmesh_token_calibration_other_per_char"][2].append(
            (labels, metrics["other_per_char"])
        )
        families["nmesh_token_calibration_samples"][2].append(
            (labels, metrics["samples"])
        )
    lines: list[str] = []
    for name, (help_text, metric_type, values) in families.items():
        lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"))
        lines.extend(
            f"{name}{_prometheus_labels(labels)} {float(value):g}"
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
        self.telemetry_keys = {
            service.name: benchmark_key(
                service.model_id, service.quant, service.backend, gpu, service.n_gpu_layers
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


def create_app(
    plan: Plan | None = None,
    watchdog: bool = False,
    watchdog_interval: float = 15.0,
) -> object:
    if FastAPI is None:
        raise ImportError("Install nmesh[gateway] to use the gateway")
    explicit = plan is not None
    selected = plan if explicit else load_plan(PLAN_PATH)
    if selected is None:
        raise FileNotFoundError("No plan found")
    @asynccontextmanager
    async def lifespan(_app: object):
        task: asyncio.Task[None] | None = None
        if watchdog:
            async def watch() -> None:
                while True:
                    await asyncio.sleep(watchdog_interval)
                    try:
                        await asyncio.to_thread(heartbeat)
                    except Exception:  # noqa: BLE001, S110
                        pass

            task = asyncio.create_task(watch())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

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
    plan_state = _PlanState(selected, explicit, gate, limiter)
    api_key = os.environ.get("NMESH_API_KEY")
    api_key_bytes = api_key.encode("utf-8") if api_key is not None else None

    @app.middleware("http")
    async def authenticate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if api_key_bytes is not None and path.startswith(("/v1/", "/metrics")):
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
        return await call_next(request)

    async def proxy(request: dict[str, object], service: PlannedService,
                    plan_snapshot: Plan, telemetry_keys: Mapping[str, str],
                    path: str, instrument: bool = True,
                    limit_slots: bool = False) -> object:
        started = time.perf_counter()
        slot_token: object | None = None
        if limit_slots:
            slot_token = await limiter.acquire(service, QUEUE_TIMEOUT)
            if slot_token is None:
                slots = max(1, service.memory.parallel_slots)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Service {service.name} is at its concurrency limit "
                        f"({slots} slots)"
                    ),
                    headers={"Retry-After": "1"},
                )
        locked = service.name in plan_snapshot.swap_group
        if locked:
            try:
                await asyncio.wait_for(
                    gate.acquire(
                        service.name,
                        lambda: ensure_running(service.name, plan_snapshot),
                    ),
                    timeout=300.0,
                )
            except asyncio.TimeoutError as error:
                raise HTTPException(status_code=504, detail="Timed out waiting for service swap") from error
        body = _upstream_body(request, service)
        url = f"{_base_url(service)}{path}"
        assert httpx is not None
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        if request.get("stream"):
            stream_options = request.get("stream_options")
            request_wants_usage = (
                isinstance(stream_options, Mapping)
                and bool(stream_options.get("include_usage"))
            )
            if service.backend == "llamacpp":
                upstream_stream_options = (
                    dict(stream_options) if isinstance(stream_options, Mapping) else {}
                )
                upstream_stream_options["include_usage"] = True
                body["stream_options"] = upstream_stream_options
            try:
                upstream_request = client.build_request("POST", url, json=body)
                try:
                    upstream = await client.send(upstream_request, stream=True)
                except httpx.ConnectError:
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
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                raise
            except httpx.HTTPError as error:
                await client.aclose()
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                raise HTTPException(status_code=502, detail=str(error)) from error
            if upstream.status_code >= 400:
                content = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                if locked:
                    gate.release()
                if limit_slots:
                    limiter.release(slot_token)
                error = _completion_not_supported(path, upstream.status_code, service.backend)
                if error is not None:
                    raise error
                return Response(content=content, status_code=upstream.status_code,
                                media_type=upstream.headers.get("content-type"))

            async def stream() -> AsyncIterator[bytes]:
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
                    if "model" in payload:
                        payload["model"] = request.get("model", service.model_id)
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
                except httpx.HTTPError as error:
                    raise HTTPException(status_code=502, detail=str(error)) from error
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    if locked:
                        gate.release()
                    if limit_slots:
                        limiter.release(slot_token)
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
                            ))
                        except Exception:  # noqa: BLE001, S110
                            pass
                    await _record_prompt_calibration(service, request, usage)
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            try:
                response = await client.post(url, json=body)
            except httpx.ConnectError:
                try:
                    await asyncio.to_thread(ensure_running, service.name, plan_snapshot)
                except Exception as error:
                    raise HTTPException(status_code=502, detail=str(error)) from error
                response = await client.post(f"{_base_url(service)}{path}", json=body)
            content = response.content
            if response.status_code >= 400:
                error = _completion_not_supported(path, response.status_code, service.backend)
                if error is not None:
                    raise error
                return Response(content=content, status_code=response.status_code,
                                media_type=response.headers.get("content-type"))
            data = json.loads(content)
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            await client.aclose()
            if locked:
                gate.release()
            if limit_slots:
                limiter.release(slot_token)
        if isinstance(data, dict) and "model" in data:
            data["model"] = request.get("model", service.model_id)
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
                ))
            except Exception:  # noqa: BLE001, S110
                pass
        await _record_prompt_calibration(
            service,
            request,
            data.get("usage") if isinstance(data, dict) else None,
        )
        return data

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

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        return {
            "services": telemetry_summary(),
            "concurrency": limiter.metrics(),
            "token_calibration": _calibration_metrics(selected.services),
        }

    @app.get("/metrics/prometheus")
    async def metrics_prometheus() -> object:
        plan_state.maybe_reload()
        selected, _ = plan_state.snapshot()
        return Response(
            _prometheus_text(limiter, selected.services),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
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

    @app.post("/v1/embeddings")
    async def embeddings(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        service = _service(selected, selected.routing.role_to_service.get("embed", ""))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/embeddings", instrument=False
        )

    return app


app = None

__all__ = ["app", "create_app", "route"]
