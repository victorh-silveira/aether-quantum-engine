from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    _attempt_post_settlement_trading_cycle,
    _record_post_settlement_incomplete,
    _run_post_settlement_retry_loop,
    run_post_settlement_breath_and_cycle,
)
from tests.unit.application.post_settlement_helpers import (
    patch_incrementing_monotonic,
    patch_instant_post_settlement_poll,
)


POST_SETTLEMENT_MODULE = "src.application.services.orchestrator.post_settlement_cycle"


def test_record_post_settlement_incomplete_does_not_set_deadlock(orch_ready):
    orch = orch_ready
    _record_post_settlement_incomplete(orch)
    assert orch._post_settlement_incomplete_streak == 1
    assert orch._post_settlement_deadlock is False
    _record_post_settlement_incomplete(orch)
    assert orch._post_settlement_incomplete_streak == 2
    assert orch._post_settlement_deadlock is False


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
    orch.config["orchestrator"]["settlement_tolerance_window_seconds"] = 1.0
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
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._await_post_settlement_breath", new_callable=AsyncMock),
        patch_incrementing_monotonic(step=0.5),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    shutdown_mock.assert_awaited()
    assert shutdown_mock.await_args.kwargs["fast_path"] is True
    assert attempts >= 1


@pytest.mark.asyncio
async def test_post_settlement_tolerance_window_runs_orphan_cleaner(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["settlement_tolerance_window_seconds"] = 1.0
    cleaner_calls = 0

    async def fake_reconcile(_self):
        nonlocal cleaner_calls
        cleaner_calls += 1
        if cleaner_calls >= 1:
            orch.running = False
        return 1

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", new_callable=AsyncMock, return_value=False),
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new=fake_reconcile,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._await_post_settlement_breath", new_callable=AsyncMock),
        patch_incrementing_monotonic(step=0.6),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    assert cleaner_calls >= 1
    assert orch._post_settlement_deadlock is False


@pytest.mark.asyncio
async def test_post_settlement_retry_loop_exits_on_stop_win(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})
    with patch(f"{POST_SETTLEMENT_MODULE}._try_stop_win_fast_path", new_callable=AsyncMock, return_value=True):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)


@pytest.mark.asyncio
async def test_post_settlement_retry_loop_soft_recovers_after_tolerance_window(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})
    orch_cfg["settlement_tolerance_window_seconds"] = 1.0
    attempts = 0

    async def cycle_without_cluster():
        nonlocal attempts
        attempts += 1
        orch._last_cycle_cluster_executed = False
        if attempts >= 3:
            orch.running = False
        return True

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", AsyncMock(side_effect=cycle_without_cluster)),
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._await_post_settlement_breath", new_callable=AsyncMock),
        patch_incrementing_monotonic(step=0.6),
        patch_instant_post_settlement_poll(),
    ):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)

    assert orch._post_settlement_deadlock is False
    assert attempts >= 2


@pytest.mark.asyncio
async def test_post_settlement_active_contracts_trigger_orphan_cleaner(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})
    orch_cfg["settlement_tolerance_window_seconds"] = 1.0
    orch.state.active_contracts = {11: object()}
    polls = 0

    async def release_contracts(_seconds: float):
        nonlocal polls
        polls += 1
        if polls >= 2:
            orch.state.active_contracts = {}
            orch.running = False

    with (
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=1,
        ) as cleaner,
        patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", side_effect=release_contracts),
        patch_incrementing_monotonic(step=0.7),
    ):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)
    cleaner.assert_awaited()


@pytest.mark.asyncio
async def test_post_settlement_recovery_then_successful_cycle(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})
    orch_cfg["settlement_tolerance_window_seconds"] = 1.0
    calls = 0

    async def cycle_then_success():
        nonlocal calls
        calls += 1
        orch._last_cycle_cluster_executed = calls >= 2
        return True

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", AsyncMock(side_effect=cycle_then_success)),
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._await_post_settlement_breath", new_callable=AsyncMock),
        patch_incrementing_monotonic(step=0.8),
        patch_instant_post_settlement_poll(),
    ):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)
    assert calls >= 2
    assert orch._post_settlement_deadlock is False


@pytest.mark.asyncio
async def test_post_settlement_cycle_requires_cluster_execution(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})

    async def cycle_without_cluster():
        orch._last_cycle_cluster_executed = False
        return True

    async def cycle_with_cluster():
        orch._last_cycle_cluster_executed = True
        return True

    orch._run_trading_cycle_if_ready = AsyncMock(side_effect=cycle_without_cluster)
    assert await _attempt_post_settlement_trading_cycle(orch, orch_cfg) is False
    orch._run_trading_cycle_if_ready = AsyncMock(side_effect=cycle_with_cluster)
    assert await _attempt_post_settlement_trading_cycle(orch, orch_cfg) is True
