from __future__ import annotations

import asyncio

from nmesh.gateway.gate import SwapGate


def test_same_service_acquires_as_concurrent_readers() -> None:
    async def scenario() -> None:
        gate = SwapGate()
        calls = 0

        def ensure() -> None:
            nonlocal calls
            calls += 1

        await asyncio.gather(
            gate.acquire("chat", ensure),
            gate.acquire("chat", ensure),
        )
        assert calls == 1
        assert gate._active == 2
        gate.release()
        gate.release()

    asyncio.run(scenario())


def test_second_service_waits_for_readers_to_drain() -> None:
    async def scenario() -> None:
        gate = SwapGate()
        sequence: list[str] = []
        await gate.acquire("chat", lambda: sequence.append("chat"))

        async def acquire_code() -> None:
            await gate.acquire(
                "code",
                lambda: sequence.append(f"code:{gate._active}"),
            )

        task = asyncio.create_task(acquire_code())
        await asyncio.sleep(0)
        assert sequence == ["chat"]
        gate.release()
        await task
        assert sequence == ["chat", "code:0"]
        gate.release()

    asyncio.run(scenario())


def test_timed_out_acquire_leaves_gate_usable() -> None:
    async def scenario() -> None:
        gate = SwapGate()
        await gate.acquire("chat", lambda: None)
        try:
            await asyncio.wait_for(
                gate.acquire("code", lambda: None),
                timeout=0.01,
            )
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("acquire unexpectedly completed")
        assert not gate._swapping
        gate.release()
        await asyncio.wait_for(gate.acquire("code", lambda: None), timeout=0.1)
        gate.release()

    asyncio.run(scenario())


def test_invalidate_forces_next_acquire_to_ensure() -> None:
    async def scenario() -> None:
        gate = SwapGate()
        calls = 0

        def ensure() -> None:
            nonlocal calls
            calls += 1

        await gate.acquire("chat", ensure)
        gate.release()
        gate.invalidate()
        await gate.acquire("chat", ensure)
        assert calls == 2
        gate.release()

    asyncio.run(scenario())


def test_failed_ensure_drops_current_so_next_acquire_re_ensures() -> None:
    """A swap whose ensure dies mid-way may have stopped the outgoing member;
    the gate must not keep fast-pathing requests for that dead incumbent."""
    async def scenario() -> None:
        gate = SwapGate()
        calls: list[str] = []
        await gate.acquire("chat", lambda: calls.append("ensure:chat"))
        gate.release()

        def broken() -> None:
            calls.append("ensure:code")
            raise RuntimeError("spawn failed")

        try:
            await gate.acquire("code", broken)
        except RuntimeError:
            pass
        else:
            raise AssertionError("ensure unexpectedly succeeded")

        # The incumbent "chat" is no longer trusted current: re-acquiring it
        # must run its ensure again instead of proxying to a dead engine.
        await gate.acquire("chat", lambda: calls.append("ensure:chat"))
        gate.release()
        assert calls == ["ensure:chat", "ensure:code", "ensure:chat"]

    asyncio.run(scenario())


def test_cancelled_acquire_drops_current() -> None:
    """wait_for cancellation while draining leaves the same stale-current
    hazard — the incumbent may have been stopped before ensure raised, so a
    cancelled swap also re-ensures on the next request."""
    async def scenario() -> None:
        gate = SwapGate()
        calls: list[str] = []
        await gate.acquire("chat", lambda: calls.append("ensure:chat"))

        try:
            await asyncio.wait_for(
                gate.acquire("code", lambda: calls.append("ensure:code")),
                timeout=0.05,
            )
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("acquire unexpectedly completed")

        gate.release()
        await gate.acquire("chat", lambda: calls.append("ensure:chat"))
        gate.release()
        assert calls == ["ensure:chat", "ensure:chat"]

    asyncio.run(scenario())
