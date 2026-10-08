"""Testes do encerramento gracioso em fast-path."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown


@pytest.mark.asyncio
async def test_graceful_shutdown_fast_path_cancels_settlement_queue():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {}}
    orch.infra = None
    orch.ws = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_settlement_queue_fast",
        ) as cancel_queue,
        patch(
            "src.application.services.orchestrator.graceful_shutdown._cancel_pending_loop_tasks",
            new_callable=AsyncMock,
        ) as cancel_tasks,
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infrastructure_connections",
            new_callable=AsyncMock,
        ) as close_infra,
        patch("src.application.services.orchestrator.graceful_shutdown.os._exit") as hard_exit,
    ):
        await graceful_shutdown(orch, fast_path=True)
    cancel_queue.assert_called_once_with(orch)
    cancel_tasks.assert_awaited_once()
    close_infra.assert_awaited_once()
    hard_exit.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_graceful_shutdown_cancels_pending_tasks_before_infra_close():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {}}
    orch.infra = None
    orch.ws = AsyncMock()
    started = asyncio.Event()

    async def _orphan() -> None:
        started.set()
        await asyncio.sleep(3600)

    orphan = asyncio.create_task(_orphan(), name="_run_settlement_watch")
    await started.wait()
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infrastructure_connections",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.graceful_shutdown.os._exit"),
    ):
        await graceful_shutdown(orch, fast_path=True)
    assert orphan.cancelled() or orphan.done()


@pytest.mark.asyncio
async def test_graceful_shutdown_non_fast_path_awaits_post_settlement_task():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch.config = {"infra": {}}
    orch.infra = None
    orch.ws = AsyncMock()

    async def _slow():
        await asyncio.sleep(10)

    task = asyncio.create_task(_slow())
    orch._post_settlement_task = task
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infrastructure_connections",
            new_callable=AsyncMock,
        ) as close_infra,
        patch(
            "src.application.services.orchestrator.graceful_shutdown._cancel_pending_loop_tasks",
            new_callable=AsyncMock,
        ) as cancel_tasks,
    ):
        await graceful_shutdown(orch, fast_path=False)
    cancel_tasks.assert_awaited_once()
    close_infra.assert_awaited_once()
    assert task.cancelled() or task.done()
