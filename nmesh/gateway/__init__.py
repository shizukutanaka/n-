from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict

from nmesh.planner import Plan, PlannedService, load_plan
from nmesh.runtime import ensure_running
from nmesh.runtime import status as runtime_status

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
    if _get(request, "tools"):
        return plan.routing.role_to_service.get("tools", plan.routing.role_to_service.get("chat", ""))
    model = _get(request, "model")
    if isinstance(model, str) and model.startswith("nmesh-code"):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    if isinstance(model, str) and model.startswith("nmesh-embed"):
        return plan.routing.role_to_service.get("embed", "")
    content = _content(request)
    if re.search(r"(?:```|^\s*(?:def |class |function |SELECT |import ))", content, re.MULTILINE):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    chat_name = plan.routing.role_to_service.get("chat", "")
    chat = next((item for item in plan.services if item.name == chat_name), None)
    if chat and len(content) // 4 > chat.context * 0.8:
        return max(plan.services, key=lambda item: item.context, default=chat).name
    return chat_name or (plan.services[0].name if plan.services else "")


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

    async def proxy(request: dict[str, object], service: PlannedService,
                    path: str) -> object:
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
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                except httpx.HTTPError as error:
                    raise HTTPException(status_code=502, detail=str(error)) from error
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    if locked:
                        swap_lock.release()
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

    @app.post("/v1/chat/completions")
    async def completions(request: dict[str, object]) -> object:
        return await proxy(request, _service(selected, route(request, selected)),
                           "/v1/chat/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: dict[str, object]) -> object:
        service = _service(selected, selected.routing.role_to_service.get("embed", ""))
        return await proxy(request, service, "/v1/embeddings")

    return app


app = None

__all__ = ["app", "create_app", "route"]
