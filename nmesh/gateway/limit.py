from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nmesh.planner import Plan, PlannedService

_UNLIMITED = object()


@dataclass
class _Entry:
    semaphore: asyncio.Semaphore
    limit: int
    in_flight: int = 0
    waiting: int = 0


class SlotLimiter:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def size(self, plan: Plan) -> None:
        self._entries = {
            service.name: _Entry(
                asyncio.Semaphore(max(1, service.memory.parallel_slots)),
                max(1, service.memory.parallel_slots),
            )
            for service in plan.services
            if service.backend in {"llamacpp", "vllm"}
        }

    async def acquire(
        self, service: PlannedService, timeout: float
    ) -> object | None:
        entry = self._entries.get(service.name)
        if entry is None:
            return _UNLIMITED
        entry.waiting += 1
        try:
            try:
                await asyncio.wait_for(entry.semaphore.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        finally:
            entry.waiting -= 1
        entry.in_flight += 1
        return entry

    def release(self, token: object | None) -> None:
        if isinstance(token, _Entry):
            token.in_flight -= 1
            token.semaphore.release()

    def metrics(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "limit": entry.limit,
                "in_flight": entry.in_flight,
                "waiting": entry.waiting,
            }
            for name, entry in self._entries.items()
        }
