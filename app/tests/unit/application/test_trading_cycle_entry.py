from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, apply_engine_mode
from src.application.services.orchestrator.trading_cycle_entry import (
    acquire_trading_cycle_lock,
    run_trading_cycle_if_ready,
    trading_cycle_entry_allowed,
)
from src.application.services.orchestrator.trading_cycle_entry_guards import _stop_win_blocks_cycle


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"
GUARDS_MODULE = "src.application.services.orchestrator.trading_cycle_entry_guards"


def test_trading_cycle_entry_blocked_in_train_engine_mode(orch_ready):
    orch = orch_ready
    apply_engine_mode(orch.config, ENGINE_MODE_TRAIN)
    assert trading_cycle_entry_allowed(orch) is False


def test_trading_cycle_entry_blocked_when_is_trading(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    assert trading_cycle_entry_allowed(orch) is False


def test_trading_cycle_entry_waits_for_settlement(orch_ready):
    orch = orch_ready
    orch.state.active_contracts = {1: object()}
    orch.logger = MagicMock()
    assert trading_cycle_entry_allowed(orch) is False
    assert orch._settlement_wait_logged is True
    orch.logger.info.assert_not_called()


def test_trading_cycle_entry_allowed(orch_ready):
    orch = orch_ready
    orch._settlement_wait_logged = True
    assert trading_cycle_entry_allowed(orch) is True
    assert orch._settlement_wait_logged is False


def test_trading_cycle_entry_allowed_when_cadence_elapsed_same_epoch(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 60
    orch._last_epoch = 1000
    orch._last_processed_epoch = 1000
    orch._last_cluster_cycle_end = 10.0
    with patch(f"{GUARDS_MODULE}.time.time", return_value=80.0):
        assert trading_cycle_entry_allowed(orch) is True


def test_trading_cycle_entry_blocked_same_epoch_before_cadence(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 60
    orch._last_epoch = 1000
    orch._last_processed_epoch = 1000
    orch._last_cluster_cycle_end = 50.0
    with patch(f"{GUARDS_MODULE}.time.time", return_value=80.0):
        assert trading_cycle_entry_allowed(orch) is False


def test_trading_cycle_entry_blocked_after_shutdown(orch_ready):
    orch = orch_ready
    orch.running = False
    orch.shutdown_reason = "stop_win"
    assert trading_cycle_entry_allowed(orch) is False


def test_stop_win_blocks_cycle_via_shutdown_reason(orch_ready):
    orch = orch_ready
    orch.shutdown_reason = "stop_win"
    assert _stop_win_blocks_cycle(orch) is True


def test_stop_win_blocks_cycle_without_risk_manager():
    orch = SimpleNamespace(shutdown_reason=None, risk_manager=None)
    assert _stop_win_blocks_cycle(orch) is False


def test_stop_win_blocks_cycle_when_target_zero(orch_ready):
    orch = orch_ready
    orch.risk_manager.total_session_profit = 999.0
    orch.config["risk_management"] = {
        "small_account_threshold": 50.0,
        "large_account_stop_win_pct": 0.0,
    }
    assert _stop_win_blocks_cycle(orch) is False


def test_trading_cycle_entry_blocked_when_stop_win_reached(orch_ready):
    orch = orch_ready
    orch.risk_manager.initial_bankroll = 1000.0
    orch.risk_manager.total_session_profit = 50.0
    orch.config["risk_management"] = {
        "small_account_threshold": 50.0,
        "large_account_stop_win_pct": 4.0,
    }
    assert trading_cycle_entry_allowed(orch) is False


@pytest.mark.asyncio
async def test_acquire_trading_cycle_lock_reserves_slot(orch_ready):
    orch = orch_ready
    assert await acquire_trading_cycle_lock(orch) is True
    assert orch.is_trading is True


@pytest.mark.asyncio
async def test_acquire_trading_cycle_lock_rejects_when_busy(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    assert await acquire_trading_cycle_lock(orch) is False


@pytest.mark.asyncio
async def test_acquire_trading_cycle_lock_rejects_when_stop_win_reached(orch_ready):
    orch = orch_ready
    orch.risk_manager.initial_bankroll = 1000.0
    orch.risk_manager.total_session_profit = 50.0
    orch.config["risk_management"] = {
        "small_account_threshold": 50.0,
        "large_account_stop_win_pct": 4.0,
    }
    assert await acquire_trading_cycle_lock(orch) is False


@pytest.mark.asyncio
async def test_trading_cycle_logs_quality_guard_reason_on_cluster_suspend(orch_ready, caplog):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.executor.execute_cluster = AsyncMock()
    weak_decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.55,
            }
        },
    }
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        caplog.at_level("INFO", logger="AETH"),
    ):
        await run_trading_cycle_if_ready(orch)
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert guard_logs
    message = guard_logs[0].message
    assert "TCN Margin" in message
    assert "<" in message
    assert "min" in message
    assert "linear=0" in message
    assert "Payoff" not in message
    assert "None" not in message


@pytest.mark.asyncio
async def test_trading_cycle_skips_epoch_advance_when_quality_guard_suspends(orch_ready):
    orch = orch_ready
    orch._last_epoch = 500
    orch._last_processed_epoch = 0
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    weak_decisions = {"RDBULL": {"metrics": {"calibrated_prob": 0.55}}}
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock) as mark_mock,
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
    ):
        await run_trading_cycle_if_ready(orch)
    mark_mock.assert_not_awaited()
    assert orch._last_processed_epoch == 0


@pytest.mark.asyncio
async def test_trading_cycle_increments_starvation_counter_on_quality_suspend(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch._quality_skipped_cycles_counter = 5
    weak_decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.55,
            }
        },
    }
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.record_quality_guard_cycle_skip") as mock_record,
    ):
        await run_trading_cycle_if_ready(orch)
    mock_record.assert_called_once_with(orch)


@pytest.mark.asyncio
async def test_trading_cycle_resets_starvation_counter_on_success(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch._quality_skipped_cycles_counter = 5
    strong_decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.90,
                "deploy_ok": True,
            }
        },
    }
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=strong_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        patch(
            f"{TRADING_CYCLE_MODULE}.reset_quality_skipped_cycles_counter_for_orch", new_callable=AsyncMock
        ) as mock_reset,
    ):
        await run_trading_cycle_if_ready(orch)
    mock_reset.assert_awaited_once_with(orch)
