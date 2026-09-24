from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from nmesh.planner import Plan, PlannedService

_UNLIMITED = object()


@dataclass
class _Entry:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    limit: int = 1
    in_flight: int = 0
    waiting: int = 0


class SlotLimiter:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def size(self, plan: Plan) -> None:
        # Entries persist across plan reloads: a request already holding a slot
        # keeps decrementing the same counter on release, so a replaced plan can
        # never admit stale + limit concurrent upstream work. Tokens still held
        # on a dropped service release into the orphaned entry, which is fine.
        wanted = {
            service.name: max(1, service.memory.parallel_slots)
            for service in plan.services
            if service.backend in {"llamacpp", "vllm"}
        }
        for name in list(self._entries):
            if name not in wanted:
                del self._entries[name]
        for name, limit in wanted.items():
            entry = self._entries.get(name)
            if entry is None:
                self._entries[name] = _Entry(asyncio.Condition(), limit)
            else:
                entry.limit = limit

    async def acquire(
        self, service: PlannedService, timeout: float
    ) -> object | None:
        entry = self._entries.get(service.name)
        if entry is None:
            return _UNLIMITED
        entry.waiting += 1
        try:
            try:
                async with entry.condition:
                    await asyncio.wait_for(
                        entry.condition.wait_for(
                            lambda: entry.in_flight < entry.limit
                        ),
                        timeout=timeout,
                    )
                    entry.in_flight += 1
            except asyncio.TimeoutError:
                return None
            return entry
        finally:
            entry.waiting -= 1

    async def release(self, token: object | None) -> None:
        if isinstance(token, _Entry):
            async with token.condition:
                token.in_flight -= 1
                token.condition.notify()

    def metrics(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "limit": entry.limit,
                "in_flight": entry.in_flight,
                "waiting": entry.waiting,
            }
            for name, entry in self._entries.items()
        }
