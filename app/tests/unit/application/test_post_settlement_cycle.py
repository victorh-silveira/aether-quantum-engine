import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    _release_post_settlement_task,
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)


def test_release_post_settlement_task_clears_matching_reference(orch_ready):
    task = MagicMock()
    orch_ready._post_settlement_task = task
    _release_post_settlement_task(orch_ready, task)
    assert orch_ready._post_settlement_task is None


def test_release_post_settlement_task_ignores_stale_callback(orch_ready):
    stale = MagicMock()
    newer = MagicMock()
    orch_ready._post_settlement_task = newer
    _release_post_settlement_task(orch_ready, stale)
    assert orch_ready._post_settlement_task is newer


def test_schedule_skips_when_not_running(orch_ready):
    orch_ready.running = False
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        schedule_trading_cycle_after_settlement(orch_ready)
    mock_create.assert_not_called()


def test_schedule_skips_when_active_contracts(orch_ready):
    orch_ready.state.active_contracts = {1: object()}
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        schedule_trading_cycle_after_settlement(orch_ready)
    mock_create.assert_not_called()


def test_schedule_skips_when_pending_task(orch_ready):
    pending = MagicMock()
    pending.done.return_value = False
    orch_ready._post_settlement_task = pending
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        schedule_trading_cycle_after_settlement(orch_ready)
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_run_post_settlement_invokes_real_trading_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await run_post_settlement_breath_and_cycle(orch)
    orch.executor.execute_cluster.assert_awaited_once()
    assert orch._post_settlement_task is None


@pytest.mark.asyncio
async def test_run_post_settlement_releases_task_reference(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock),
    ):
        orch.executor.execute_cluster = AsyncMock()
        task = asyncio.create_task(run_post_settlement_breath_and_cycle(orch))
        orch._post_settlement_task = task
        await task
    assert orch._post_settlement_task is None


@pytest.mark.asyncio
async def test_run_post_settlement_waits_for_is_trading_then_runs_real_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 2.0
    orch.is_trading = True

    async def release_trading():
        await asyncio.sleep(0.35)
        orch.is_trading = False

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await asyncio.gather(run_post_settlement_breath_and_cycle(orch), release_trading())

    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_post_settlement_reschedules_when_is_trading_timeout(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 0.01
    times = iter([0.0, 0.02, 0.04])

    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.time.monotonic",
            side_effect=lambda: next(times, 1.0),
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create,
    ):
        await run_post_settlement_breath_and_cycle(orch)

    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_spawns_task_even_when_is_trading(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock),
    ):
        orch.executor.execute_cluster = AsyncMock()
        schedule_trading_cycle_after_settlement(orch)
        assert orch._post_settlement_task is not None
        orch.is_trading = False
        await orch._post_settlement_task
    orch.executor.execute_cluster.assert_awaited_once()
