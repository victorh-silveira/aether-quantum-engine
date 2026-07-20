import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_settlement import _settlement_poll_delay
from src.application.services.orchestrator.post_settlement_cycle import (
    _await_post_settlement_breath,
    _clean_stale_settlement_and_redis_counters,
    _ensure_trading_slot_poll,
    _poll_delay,
    _prune_stale_risk_contract_ids,
    _record_post_settlement_incomplete,
    _release_post_settlement_task,
    _release_stuck_trading_slot,
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)
from tests.unit.application.post_settlement_helpers import (
    TRADING_CYCLE_COLLECT,
    patch_incrementing_monotonic,
    patch_instant_post_settlement_poll,
)


POST_SETTLEMENT_MODULE = "src.application.services.orchestrator.post_settlement_cycle"


def test_release_post_settlement_task_clears_matching_reference(orch_ready):
    task = MagicMock()
    orch_ready._post_settlement_task = task
    _release_post_settlement_task(orch_ready, task)
    assert orch_ready._post_settlement_task is None


def test_prune_stale_risk_contract_ids_clears_orphans(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = [888, 999]
    orch.risk_manager.contract_to_symbol[888] = "R_10"
    orch.risk_manager.contract_to_symbol[999] = "R_10"
    orch.risk_manager.cluster_results[999] = -1.0
    _prune_stale_risk_contract_ids(orch)
    assert orch.risk_manager.active_contract_ids == []
    assert 999 not in orch.risk_manager.contract_to_symbol
    assert 999 not in orch.risk_manager.cluster_results
    assert 888 not in orch.risk_manager.contract_to_symbol


def test_release_post_settlement_task_ignores_stale_callback(orch_ready):
    stale = MagicMock()
    newer = MagicMock()
    orch_ready._post_settlement_task = newer
    _release_post_settlement_task(orch_ready, stale)
    assert orch_ready._post_settlement_task is newer


@pytest.mark.asyncio
async def test_release_stuck_trading_slot_polls_before_deadline(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    times = iter([0.0, 0.0, 10.0])
    with (
        patch(f"{POST_SETTLEMENT_MODULE}.time.monotonic", side_effect=lambda: next(times, 10.0)),
        patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", new_callable=AsyncMock) as poll_mock,
    ):
        await _release_stuck_trading_slot(orch, wait_limit=5.0)
    poll_mock.assert_awaited()
    assert orch.is_trading is False


def test_ensure_trading_slot_poll_skips_active_task(orch_ready):
    orch = orch_ready
    pending = asyncio.get_event_loop().create_future()
    orch._trading_slot_poll_task = pending
    _ensure_trading_slot_poll(orch, wait_limit=1.0)
    assert orch._trading_slot_poll_task is pending
    pending.cancel()


@pytest.mark.asyncio
async def test_poll_delay_yields():
    await _poll_delay(0)


@pytest.mark.asyncio
async def test_settlement_poll_delay_yields():
    await _settlement_poll_delay(0)


@pytest.mark.asyncio
async def test_breath_completes_after_timeout_slices(orch_ready):
    orch = orch_ready
    await _await_post_settlement_breath(orch, 0.5, 0.2)
    assert not orch._post_settlement_wake.is_set()


@pytest.mark.asyncio
async def test_breath_interrupts_on_wake(orch_ready):
    orch = orch_ready
    orch._post_settlement_wake.clear()
    task = asyncio.create_task(_await_post_settlement_breath(orch, 5.0, 0.25))
    await asyncio.sleep(0)
    orch._post_settlement_wake.set()
    await task


@pytest.mark.asyncio
async def test_breath_returns_when_wake_completes_before_timeout(orch_ready):
    orch = orch_ready
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle.asyncio.wait_for",
        new_callable=AsyncMock,
    ):
        await _await_post_settlement_breath(orch, 1.0, 0.5)


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
    orch_ready._post_settlement_wake.clear()
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        schedule_trading_cycle_after_settlement(orch_ready)
    mock_create.assert_not_called()
    assert orch_ready._post_settlement_wake.is_set()


@pytest.mark.asyncio
async def test_run_post_settlement_invokes_real_trading_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    with (
        patch(
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
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
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
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
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 0.01
    orch.is_trading = True

    with (
        patch(
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_incrementing_monotonic(),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await run_post_settlement_breath_and_cycle(orch)

    orch.executor.execute_cluster.assert_awaited_once()
    assert orch.is_trading is False


@pytest.mark.asyncio
async def test_clean_stale_settlement_and_redis_counters_execution(orch_ready):
    orch = orch_ready
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_redis.pipeline.return_value = mock_pipe

    async def mock_redis_func():
        return mock_redis

    orch.state_store = MagicMock()
    orch.state_store._redis = mock_redis_func

    await _clean_stale_settlement_and_redis_counters(orch)

    mock_pipe.delete.assert_any_call("recovery:skip_counter")
    mock_pipe.delete.assert_any_call("settlement:queue:priority")
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_post_settlement_incomplete_with_loop(orch_ready):
    orch = orch_ready
    with patch(
        f"{POST_SETTLEMENT_MODULE}._clean_stale_settlement_and_redis_counters",
        new_callable=AsyncMock,
    ) as mock_clean:
        _record_post_settlement_incomplete(orch)
        await asyncio.sleep(0.01)
        mock_clean.assert_called_once_with(orch)
