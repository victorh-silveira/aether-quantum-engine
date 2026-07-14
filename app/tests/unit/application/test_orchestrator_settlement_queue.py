import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_settlement_queue import (
    _settlement_worker_loop,
    enqueue_contract_settlement,
    start_settlement_worker,
)
from src.application.services.orchestrator.settlement_queue_ops import cancel_settlement_queue_fast


real_sleep = asyncio.sleep


async def _stop_settlement_worker(orch) -> None:
    orch.running = False
    task = getattr(orch, "_settlement_worker_task", None)
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_enqueue_contract_settlement_uses_queue(orch_ready):
    orch = orch_ready
    orch.running = True
    await start_settlement_worker(orch)
    try:
        with (
            patch(
                "src.application.services.orchestrator.orchestrator_settlement_queue.process_contract_settlement",
                new_callable=AsyncMock,
            ) as settle_mock,
            patch(
                "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
                new_callable=AsyncMock,
            ),
        ):
            await enqueue_contract_settlement(orch, {"proposal_open_contract": {"contract_id": 1}})
            await asyncio.wait_for(orch._settlement_queue.join(), timeout=1.0)
        settle_mock.assert_awaited_once()
    finally:
        await _stop_settlement_worker(orch)


@pytest.mark.asyncio
async def test_enqueue_contract_settlement_fallback_without_worker(orch_ready):
    orch = orch_ready
    orch._settlement_queue = None
    with patch(
        "src.application.services.orchestrator.orchestrator_settlement_queue.process_contract_settlement",
        new_callable=AsyncMock,
    ) as settle_mock:
        await enqueue_contract_settlement(orch, {"proposal_open_contract": {"contract_id": 2}})
    settle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_settlement_worker_is_idempotent(orch_ready):
    orch = orch_ready
    orch.running = True
    await start_settlement_worker(orch)
    try:
        first = orch._settlement_worker_task
        await start_settlement_worker(orch)
        assert orch._settlement_worker_task is first
    finally:
        await _stop_settlement_worker(orch)


@pytest.mark.asyncio
async def test_settlement_worker_continues_after_queue_timeout(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._settlement_queue = asyncio.Queue()

    with (
        patch("asyncio.sleep", new=real_sleep),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
            new_callable=AsyncMock,
        ),
    ):
        task = asyncio.create_task(_settlement_worker_loop(orch))

        await real_sleep(0.35)

        orch.running = False
        await task


@pytest.mark.asyncio
async def test_cancel_settlement_queue_fast_drains_without_handshake(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._settlement_queue = asyncio.Queue()
    orch._settlement_queue.put_nowait({"proposal_open_contract": {"contract_id": 1}})
    orch._settlement_queue.put_nowait({"proposal_open_contract": {"contract_id": 2}})

    async def _blocked_worker():
        await asyncio.sleep(3600)

    orch._settlement_worker_task = asyncio.create_task(_blocked_worker())
    cancel_settlement_queue_fast(orch)
    assert orch._settlement_queue.empty()
    assert orch._settlement_worker_task is None
