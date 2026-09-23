from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, apply_engine_mode
from src.application.services.orchestrator.trading_cycle_entry import (
    acquire_trading_cycle_lock,
    trading_cycle_entry_allowed,
)
from src.application.services.orchestrator.trading_cycle_entry_guards import _stop_win_blocks_cycle


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


def test_trading_cycle_entry_allowed_when_cadence_zero_baseline(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 60
    orch._last_cluster_cycle_end = 0.0
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
