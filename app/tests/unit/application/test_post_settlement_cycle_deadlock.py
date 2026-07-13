from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    _MAX_POST_SETTLEMENT_CYCLE_ATTEMPTS,
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


@pytest.mark.asyncio
async def test_post_settlement_retry_loop_deadlocks_after_repeated_failed_attempt_batches(orch_ready):
    orch = orch_ready
    orch_cfg = orch.config.setdefault("orchestrator", {})
    orch_cfg["post_settlement_cycle_retry_seconds"] = 9999.0
    attempts = 0

    async def cycle_without_cluster():
        nonlocal attempts
        attempts += 1
        orch._last_cycle_cluster_executed = False
        return True

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", AsyncMock(side_effect=cycle_without_cluster)),
        patch_instant_post_settlement_poll(),
    ):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.0)

    assert orch._post_settlement_deadlock is True
    assert attempts >= _MAX_POST_SETTLEMENT_CYCLE_ATTEMPTS * 2


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
