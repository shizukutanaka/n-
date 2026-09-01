from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict

from nmesh.bench import benchmark_key
from nmesh.planner import Plan, PlannedService, load_plan
from nmesh.runtime import ensure_running
from nmesh.runtime import status as runtime_status
from nmesh.telemetry import Sample
from nmesh.telemetry import record as record_telemetry
from nmesh.telemetry import summary as telemetry_summary

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
    if not isinstance(messages, list):
        return ""
    return " ".join(
        str(item.get("content", "")) for item in messages if isinstance(item, Mapping)
    )


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
    if chat and len(content) // 4 > chat.context * 0.8:
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


def create_app(plan: Plan | None = None) -> object:
    if FastAPI is None:
        raise ImportError("Install nmesh[gateway] to use the gateway")
    selected = plan or load_plan()
    if selected is None:
        raise FileNotFoundError("No plan found")
    app = FastAPI(title="nmesh gateway")
    swap_lock = asyncio.Lock()
    gpu = selected.profile.gpus[0].name if selected.profile.gpus else "cpu"
    telemetry_keys = {
        service.name: benchmark_key(service.model_id, service.quant, service.backend,
                                    gpu, service.n_gpu_layers)
        for service in selected.services
    }

    async def proxy(request: dict[str, object], service: PlannedService,
                    path: str, instrument: bool = True) -> object:
        started = time.perf_counter()
        locked = service.name in selected.swap_group
        if locked:
            try:
                await asyncio.wait_for(swap_lock.acquire(), timeout=300.0)
                await asyncio.to_thread(ensure_running, service.name, selected)
            except asyncio.TimeoutError as error:
                raise HTTPException(status_code=504, detail="Timed out waiting for service swap") from error
            except Exception:
                swap_lock.release()
                raise
        body = _upstream_body(request, service)
        url = f"{_base_url(service)}{path}"
        assert httpx is not None
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        if request.get("stream"):
            try:
                upstream = await client.send(
                    client.build_request("POST", url, json=body), stream=True
                )
            except httpx.HTTPError as error:
                await client.aclose()
                if locked:
                    swap_lock.release()
                raise HTTPException(status_code=502, detail=str(error)) from error
            if upstream.status_code >= 400:
                content = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                if locked:
                    swap_lock.release()
                return Response(content=content, status_code=upstream.status_code,
                                media_type=upstream.headers.get("content-type"))

            async def stream() -> AsyncIterator[bytes]:
                buffer = b""
                tokens = 0
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
                        swap_lock.release()
                    if instrument:
                        span = (last_line_time - first_line_time
                                if first_line_time is not None and last_line_time is not None
                                else 0.0)
                        decode = (tokens - 1) / span if tokens >= 16 and span > 0 else None
                        try:
                            await asyncio.to_thread(record_telemetry, Sample(
                                service.name, telemetry_keys[service.name], decode,
                                first_line_time - started if first_line_time is not None else None,
                                time.perf_counter() - started, tokens, time.time(),
                            ))
                        except Exception:  # noqa: BLE001, S110
                            pass
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            response = await client.post(url, json=body)
            content = response.content
            if response.status_code >= 400:
                await client.aclose()
                return Response(content=content, status_code=response.status_code,
                                media_type=response.headers.get("content-type"))
            data = json.loads(content)
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            await client.aclose()
            raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            if locked:
                swap_lock.release()
        await client.aclose()
        if isinstance(data, dict) and "model" in data:
            data["model"] = request.get("model", data["model"])
        if instrument:
            usage = data.get("usage") if isinstance(data, dict) else None
            completion_tokens = (
                int(usage.get("completion_tokens", 0))
                if isinstance(usage, dict) else 0
            )
            try:
                await asyncio.to_thread(record_telemetry, Sample(
                    service.name, telemetry_keys[service.name], None, None,
                    time.perf_counter() - started, completion_tokens, time.time(),
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
        ids = ["nmesh-auto"] + [f"nmesh-{service.name}" for service in selected.services]
        return {"object": "list", "data": [
            {"id": item, "object": "model", "owned_by": "nmesh"} for item in ids
        ]}

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return {"services": telemetry_summary()}

    @app.post("/v1/chat/completions")
    async def completions(request: dict[str, object]) -> object:
        return await proxy(request, _service(selected, route(request, selected)),
                           "/v1/chat/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: dict[str, object]) -> object:
        service = _service(selected, selected.routing.role_to_service.get("embed", ""))
        return await proxy(request, service, "/v1/embeddings", instrument=False)

    return app


app = None

__all__ = ["app", "create_app", "route"]
