import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_settlement import _settlement_poll_delay
from src.application.services.orchestrator.post_settlement_cycle import (
    _await_post_settlement_breath,
    _ensure_trading_slot_poll,
    _poll_delay,
    _prune_stale_risk_contract_ids,
    _record_post_settlement_incomplete,
    _release_post_settlement_task,
    _release_stuck_trading_slot,
    _run_post_settlement_retry_loop,
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
    orch.risk_manager.contract_to_symbol[888] = "RDBULL"
    orch.risk_manager.contract_to_symbol[999] = "RDBEAR"
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


def test_record_post_settlement_incomplete_sets_deadlock_at_limit(orch_ready):
    orch = orch_ready
    _record_post_settlement_incomplete(orch)
    assert orch._post_settlement_incomplete_streak == 1
    assert orch._post_settlement_deadlock is False
    _record_post_settlement_incomplete(orch)
    assert orch._post_settlement_incomplete_streak == 2
    assert orch._post_settlement_deadlock is True


@pytest.mark.asyncio
async def test_post_settlement_stop_win_fast_path_skips_heavy_cycle(orch_ready):
    orch = orch_ready
    orch.state_mgr.reset_session_metrics(1000.0, 50.0)
    orch.state.balance = 1060.0
    orch.state_mgr.state.total_trades_today = 2
    orch.risk_manager.total_session_profit = 60.0
    cycle_mock = AsyncMock(return_value=False)
    with (
        patch(
            f"{POST_SETTLEMENT_MODULE}.graceful_shutdown",
            new_callable=AsyncMock,
        ) as shutdown_mock,
        patch(
            f"{POST_SETTLEMENT_MODULE}.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ) as redis_clear_mock,
        patch.object(orch, "_run_trading_cycle_if_ready", cycle_mock),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    redis_clear_mock.assert_awaited_once_with(orch)
    shutdown_mock.assert_awaited_once()
    assert shutdown_mock.await_args.kwargs["fast_path"] is True
    cycle_mock.assert_not_awaited()
    assert orch.shutdown_reason == "stop_win"


@pytest.mark.asyncio
async def test_post_settlement_stop_win_during_stuck_retry_loop(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_cycle_retry_seconds"] = 0.001
    orch.state_mgr.reset_session_metrics(1000.0, 50.0)
    orch.state_mgr.state.total_trades_today = 1
    attempts = 0

    async def cycle_never_completes():
        nonlocal attempts
        attempts += 1
        if attempts >= 1:
            orch.state.balance = 1060.0
            orch.risk_manager.total_session_profit = 60.0
            orch.state_mgr.state.total_trades_today = 1
        return False

    with (
        patch(
            f"{POST_SETTLEMENT_MODULE}.graceful_shutdown",
            new_callable=AsyncMock,
        ) as shutdown_mock,
        patch.object(orch, "_run_trading_cycle_if_ready", side_effect=cycle_never_completes),
        patch_incrementing_monotonic(),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    shutdown_mock.assert_awaited()
    assert shutdown_mock.await_args.kwargs["fast_path"] is True
    assert attempts >= 1


@pytest.mark.asyncio
async def test_post_settlement_deadlock_flag_after_two_incomplete_cycles(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_cycle_retry_seconds"] = 0.001
    with (
        patch.object(orch, "_run_trading_cycle_if_ready", new_callable=AsyncMock, return_value=False),
        patch_incrementing_monotonic(),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    assert orch._post_settlement_deadlock is True
    assert orch._post_settlement_incomplete_streak == 2


@pytest.mark.asyncio
async def test_post_settlement_retry_loop_exits_on_deadlock_flag(orch_ready):
    orch = orch_ready
    orch._post_settlement_deadlock = True
    orch_cfg = orch.config.setdefault("orchestrator", {})
    with patch(f"{POST_SETTLEMENT_MODULE}._try_stop_win_fast_path", new_callable=AsyncMock, return_value=False):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)
