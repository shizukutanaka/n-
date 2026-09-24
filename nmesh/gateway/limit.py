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
        previous = self._entries
        entries: dict[str, _Entry] = {}
        for service in plan.services:
            if service.backend not in {"llamacpp", "vllm"}:
                continue
            limit = max(1, service.memory.parallel_slots)
            old = previous.get(service.name)
            if old is not None and old.in_flight > 0 and old.limit == limit:
                # In-flight requests still hold this semaphore — replacing
                # it would let new acquisitions ignore their occupancy and
                # transiently oversubscribe the backend's slots after a
                # same-limit replan under load.
                entries[service.name] = old
            else:
                entries[service.name] = _Entry(asyncio.Semaphore(limit), limit)
        self._entries = entries

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
