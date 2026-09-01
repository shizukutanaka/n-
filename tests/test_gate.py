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
