from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import asdict

from nmesh.planner import Plan, load_plan
from nmesh.runtime import status as runtime_status

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError:
    FastAPI = None
    HTTPException = RuntimeError
    StreamingResponse = None


def _get(request: Mapping[str, object], key: str, default: object = None) -> object:
    return request.get(key, default)


def _content(request: Mapping[str, object]) -> str:
    messages = _get(request, "messages", [])
    if not isinstance(messages, list):
        return ""
    return " ".join(
        str(item.get("content", ""))
        for item in messages
        if isinstance(item, Mapping)
    )


def route(request: Mapping[str, object], plan: Plan) -> str:
    tools = _get(request, "tools")
    if tools:
        return plan.routing.role_to_service.get(
            "tools", plan.routing.role_to_service.get("chat", "")
        )
    model = _get(request, "model")
    if isinstance(model, str) and model.startswith("nmesh-code"):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    if isinstance(model, str) and model.startswith("nmesh-embed"):
        return plan.routing.role_to_service.get("embed", "")
    content = _content(request)
    if re.search(r"(?:```|^\s*(?:def |class |function |SELECT |import ))", content, re.MULTILINE):
        return plan.routing.role_to_service.get("code", plan.routing.role_to_service.get("chat", ""))
    chat_name = plan.routing.role_to_service.get("chat", "")
    chat = next((service for service in plan.services if service.name == chat_name), None)
    if chat and len(content) // 4 > chat.context * 0.8:
        return max(plan.services, key=lambda item: item.context, default=chat).name
    return chat_name or (plan.services[0].name if plan.services else "")


def create_app(plan: Plan | None = None) -> object:
    if FastAPI is None:
        raise ImportError("Install nmesh[gateway] to use the gateway")
    selected = plan or load_plan()
    if selected is None:
        raise FileNotFoundError("No plan found")
    app = FastAPI(title="nmesh gateway")
    swap_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status_endpoint() -> dict[str, object]:
        return asdict(runtime_status())

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        ids = ["nmesh-auto"] + [f"nmesh-{service.name}" for service in selected.services]
        return {
            "object": "list",
            "data": [{"id": item, "object": "model", "owned_by": "nmesh"} for item in ids],
        }

    async def response_for(request: dict[str, object]) -> dict[str, object]:
        service_name = route(request, selected)
        service = next((item for item in selected.services if item.name == service_name), None)
        if service is None:
            raise HTTPException(status_code=404, detail="No service")
        try:
            await asyncio.wait_for(swap_lock.acquire(), timeout=300.0)
        except asyncio.TimeoutError as error:
            raise HTTPException(status_code=504, detail="Timed out waiting for service swap") from error
        try:
            return {
                "id": "nmesh-response",
                "object": "chat.completion",
                "model": request.get("model", "nmesh-auto"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
            }
        finally:
            swap_lock.release()

    @app.post("/v1/chat/completions")
    async def completions(request: dict[str, object]) -> object:
        result = await response_for(request)
        if request.get("stream"):
            async def events() -> object:
                yield f"data: {json.dumps(result)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return result

    @app.post("/v1/embeddings")
    async def embeddings(request: dict[str, object]) -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [], "index": 0}],
            "model": request.get("model", "nmesh-embed"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app


app = None

__all__ = ["app", "create_app", "route"]
