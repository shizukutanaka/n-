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

    async def acquire(self, name: str, ensure: Callable[[], None]) -> None:
        while True:
            if self._current == name and not self._swapping:
                self._active += 1
                return
            async with self._swap_lock:
                if self._current == name and not self._swapping:
                    continue
                self._swapping = True
                try:
                    while self._active > 0:
                        self._drained.clear()
                        await self._drained.wait()
                    await asyncio.to_thread(ensure)
                    self._current = name
                    self._active += 1
                    return
                except BaseException:
                    # A failed or cancelled ensure may have stopped the
                    # outgoing member — a stale _current would fast-path the
                    # next request for it to a dead engine. Drop the pointer
                    # so that request re-ensures; ensure_running short-
                    # circuits when the member is still alive.
                    self._current = None
                    raise
                finally:
                    self._swapping = False

    def release(self) -> None:
        self._active -= 1
        if self._active == 0:
            self._drained.set()

    def invalidate(self) -> None:
        self._current = None
