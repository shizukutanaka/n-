from __future__ import annotations

import asyncio
from collections.abc import Callable


class SwapGate:
    def __init__(self) -> None:
        self._swap_lock = asyncio.Lock()
        self._drained = asyncio.Event()
        self._drained.set()
        self._current: str | None = None
        self._active = 0
        self._swapping = False
        self._ensure_task: asyncio.Task[None] | None = None
        self._ensure_name: str | None = None

    async def acquire(self, name: str, ensure: Callable[[], None]) -> None:
        while True:
            if self._current == name and not self._swapping:
                self._active += 1
                return
            async with self._swap_lock:
                if self._current == name and not self._swapping:
                    continue
                if self._swapping:
                    ensure_task = self._ensure_task
                    if ensure_task is None:
                        continue
                else:
                    self._swapping = True
                    try:
                        while self._active > 0:
                            self._drained.clear()
                            await self._drained.wait()
                        ensure_task = asyncio.create_task(asyncio.to_thread(ensure))
                        self._ensure_task = ensure_task
                        self._ensure_name = name
                        ensure_task.add_done_callback(self._finish_ensure)
                    except BaseException:
                        self._current = None
                        self._swapping = False
                        raise
            try:
                await asyncio.shield(ensure_task)
            except BaseException:
                self._finish_ensure(ensure_task)
                raise
            self._finish_ensure(ensure_task)

    def _finish_ensure(self, task: asyncio.Task[None]) -> None:
        if task is not self._ensure_task:
            return
        if not task.done():
            return
        self._ensure_task = None
        name = self._ensure_name
        self._ensure_name = None
        try:
            error = task.exception()
        except asyncio.CancelledError:
            error = asyncio.CancelledError()
        self._current = name if error is None else None
        self._swapping = False

    def release(self) -> None:
        self._active -= 1
        if self._active == 0:
            self._drained.set()

    def invalidate(self) -> None:
        self._current = None
