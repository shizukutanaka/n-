from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict

from nmesh.bench import benchmark_key
from nmesh.planner import PLAN_PATH, Plan, PlannedService, load_plan
from nmesh.runtime import ensure_running, heartbeat
from nmesh.runtime import status as runtime_status
from nmesh.telemetry import Sample, summary_by_approximate
from nmesh.telemetry import record as record_telemetry
from nmesh.telemetry import summary as telemetry_summary

from .gate import SwapGate
from .limit import SlotLimiter

try:
    QUEUE_TIMEOUT = float(os.environ.get("NMESH_QUEUE_TIMEOUT", "120.0"))
except ValueError:
    QUEUE_TIMEOUT = 120.0

try:
    import httpx
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, StreamingResponse
except ImportError:
    FastAPI = None
    httpx = None
    HTTPException = RuntimeError
    Response = None
    StreamingResponse = None


def _get(request: Mapping[str, object], key: str, default: object = None) -> object:
    return request.get(key, default)


def _content(request: Mapping[str, object]) -> str:
    messages = _get(request, "messages", [])
    if isinstance(messages, list) and messages:
        return " ".join(
            str(item.get("content", "")) for item in messages if isinstance(item, Mapping)
        )
    prompt = _get(request, "prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return " ".join(item for item in prompt if isinstance(item, str))
    return ""


def estimate_tokens(text: str) -> int:
    cjk = sum(
        0x3000 <= ord(char) <= 0x30FF
        or 0x3400 <= ord(char) <= 0x4DBF
        or 0x4E00 <= ord(char) <= 0x9FFF
        or 0xF900 <= ord(char) <= 0xFAFF
        or 0xAC00 <= ord(char) <= 0xD7AF
        or 0xFF00 <= ord(char) <= 0xFFEF
        for char in text
    )
    return cjk + (len(text) - cjk + 3) // 4


def route(request: Mapping[str, object], plan: Plan) -> str:
    explicit = _explicit(_get(request, "model"), plan)
    if explicit is not None:
        return explicit
    if _get(request, "tools"):
        return plan.routing.role_to_service.get("tools", plan.routing.role_to_service.get("chat", ""))
    content = _content(request)
    if re.search(r"(?:```|^\s*(?:def |class |function |SELECT |import ))", content, re.MULTILINE):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    chat_name = plan.routing.role_to_service.get("chat", "")
    chat = next((item for item in plan.services if item.name == chat_name), None)
    max_tokens = request.get("max_tokens") or 0
    try:
        reserved = int(max_tokens)
    except (TypeError, ValueError):
        reserved = 0
    if chat and estimate_tokens(content) + reserved > chat.context * 0.8:
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


def _prometheus_text(limiter: SlotLimiter) -> str:
    families: dict[str, tuple[str, str, list[tuple[Mapping[str, object], object]]]] = {
        "nmesh_telemetry_samples": (
            "Number of telemetry samples.",
            "gauge",
            [],
        ),
        "nmesh_telemetry_decode_tps_median": (
            "Median measured or approximate decode throughput in tokens per second.",
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
    }
    for service, groups in summary_by_approximate().items():
        for approximate, metrics in groups.items():
            labels = {"approximate": str(approximate).lower(), "service": service}
            families["nmesh_telemetry_samples"][2].append((labels, metrics["samples"]))
            metric_names = {
                "decode_tps_median": "nmesh_telemetry_decode_tps_median",
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
    gate = SwapGate()
    limiter = SlotLimiter()
    plan_state = _PlanState(selected, explicit, gate, limiter)
    api_key = os.environ.get("NMESH_API_KEY")

    @app.middleware("http")
    async def authenticate(request: object, call_next: object) -> object:
        path = request.url.path
        if api_key is not None and path.startswith(("/v1/", "/metrics")):
            authorization = request.headers.get("authorization", "")
            prefix = "Bearer "
            presented = authorization[len(prefix):] if authorization.startswith(prefix) else ""
            if not secrets.compare_digest(presented, api_key):
                return Response(
                    content=json.dumps({"detail": "Invalid or missing API key"}),
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
                if upstream.status_code == 404 and path == "/v1/completions":
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"Backend {service.backend} does not support "
                            "/v1/completions"
                        ),
                    )
                return Response(content=content, status_code=upstream.status_code,
                                media_type=upstream.headers.get("content-type"))

            async def stream() -> AsyncIterator[bytes]:
                buffer = b""
                tokens = 0
                usage: dict[str, object] | None = None
                first_line_time: float | None = None
                last_line_time: float | None = None
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.rstrip(b"\r")
                            if line.startswith(b"data: ") and line != b"data: [DONE]":
                                try:
                                    payload = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    payload = {}
                                candidate_usage = (
                                    payload.get("usage")
                                    if isinstance(payload, dict) else None
                                )
                                if isinstance(candidate_usage, dict):
                                    usage = candidate_usage
                                    choices = payload.get("choices")
                                    if isinstance(choices, list) and not choices:
                                        continue
                                now = time.perf_counter()
                                tokens += 1
                                first_line_time = first_line_time or now
                                last_line_time = now
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
                        exact = (
                            request_wants_usage
                            and
                            usage is not None
                            and "completion_tokens" in usage
                        )
                        completion_tokens = (
                            int(usage["completion_tokens"]) if exact
                            else tokens
                        )
                        decode = (
                            max(completion_tokens - 1, 0) / span
                            if completion_tokens >= 16 and span > 0 else None
                        )
                        try:
                            await asyncio.to_thread(record_telemetry, Sample(
                                service.name, telemetry_keys[service.name], decode,
                                first_line_time - started if first_line_time is not None else None,
                                time.perf_counter() - started, completion_tokens, time.time(),
                                not exact,
                            ))
                        except Exception:  # noqa: BLE001, S110
                            pass
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
                if response.status_code == 404 and path == "/v1/completions":
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"Backend {service.backend} does not support "
                            "/v1/completions"
                        ),
                    )
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
            data["model"] = request.get("model", data["model"])
        if instrument:
            usage = data.get("usage") if isinstance(data, dict) else None
            completion_tokens = (
                int(usage.get("completion_tokens", 0))
                if isinstance(usage, dict) else 0
            )
            exact = isinstance(usage, dict) and "completion_tokens" in usage
            try:
                await asyncio.to_thread(record_telemetry, Sample(
                    service.name, telemetry_keys[service.name], None, None,
                    time.perf_counter() - started, completion_tokens, time.time(),
                    not exact,
                ))
            except Exception:  # noqa: BLE001, S110
                pass
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
        return {"object": "list", "data": [
            {"id": item, "object": "model", "owned_by": "nmesh"} for item in ids
        ]}

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
        return {
            "services": telemetry_summary(),
            "concurrency": limiter.metrics(),
        }

    @app.get("/metrics/prometheus")
    async def metrics_prometheus() -> object:
        return Response(_prometheus_text(limiter), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        service = _service(selected, route(request, selected))
        return await proxy(
            request, service, selected, telemetry_keys, "/v1/chat/completions",
            limit_slots=True,
        )

    @app.post("/v1/completions")
    async def legacy_completions(request: dict[str, object]) -> object:
        plan_state.maybe_reload()
        selected, telemetry_keys = plan_state.snapshot()
        service = _service(selected, route(request, selected))
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
