"""Testes de bootstrap de meta por sessao ativa."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.session_target_bootstrap import (
    bootstrap_active_session_targets,
    clear_current_session_redis_keys,
    current_session_redis_payload,
    restore_current_session_targets,
)
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.state.state_manager import StateManager


@pytest.mark.asyncio
async def test_bootstrap_active_session_targets():
    orch = MagicMock()
    orch._session_targets_bootstrapped = False
    orch.config = {"risk_management": {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.01}}}
    orch.state_mgr = StateManager()
    orch.risk_manager = RiskManager(orch.config["risk_management"])
    orch.state_store = AsyncMock()
    orch.logger = MagicMock()
    await bootstrap_active_session_targets(orch, 10000.0)
    assert orch.state_mgr.state.daily_stop_win_target == pytest.approx(100.0)
    assert orch.risk_manager.initial_bankroll == pytest.approx(10000.0)
    assert orch._session_targets_bootstrapped is True
    orch.state_store.set_string.assert_called()


@pytest.mark.asyncio
async def test_bootstrap_idempotent_second_call():
    orch = MagicMock()
    orch.config = {"risk_management": {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.01}}}
    orch.state_mgr = StateManager()
    orch.risk_manager = RiskManager(orch.config["risk_management"])
    orch.state_store = AsyncMock()
    orch.logger = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr.state.daily_stop_win_target = 50.0
    await bootstrap_active_session_targets(orch, 20000.0)
    assert orch.state_mgr.state.daily_stop_win_target == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_restore_current_session_targets_when_bootstrapped():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr = StateManager()
    orch.risk_manager = RiskManager({"params": {}})
    orch.state_store = AsyncMock()

    async def _get_string(key: str):
        if key == "session:current:start_balance":
            return "5000.0"
        if key == "session:current:target_win":
            return "50.0"
        return None

    orch.state_store.get_string.side_effect = _get_string
    await restore_current_session_targets(orch)
    assert orch.state_mgr.state.initial_balance == pytest.approx(5000.0)
    assert orch.risk_manager.daily_stop_win_target == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_clear_current_session_redis_keys():
    orch = MagicMock()
    orch.state_store = AsyncMock()
    await clear_current_session_redis_keys(orch)
    assert orch.state_store.delete_string.await_count == 2


def test_current_session_redis_payload():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr = StateManager()
    orch.state_mgr.state.initial_balance = 1000.0
    orch.state_mgr.state.daily_stop_win_target = 10.0
    assert current_session_redis_payload(orch) == (1000.0, 10.0)
    orch._session_targets_bootstrapped = False
    assert current_session_redis_payload(orch) == (None, None)


@pytest.mark.asyncio
async def test_bootstrap_legacy_stop_win_without_compounding():
    orch = MagicMock()
    orch._session_targets_bootstrapped = False
    orch.config = {
        "risk_management": {
            "small_account_threshold": 100.0,
            "small_account_stop_win": 15.0,
            "params": {"compounding_enabled": False},
        }
    }
    orch.state_mgr = StateManager()
    orch.risk_manager = RiskManager(orch.config["risk_management"])
    orch.state_store = AsyncMock()
    orch.logger = MagicMock()
    await bootstrap_active_session_targets(orch, 50.0)
    assert orch.state_mgr.state.daily_stop_win_target == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_restore_current_session_invalid_values():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr = StateManager()
    orch.state_store = AsyncMock()
    orch.state_store.get_string.return_value = "bad"
    await restore_current_session_targets(orch)
    assert orch.state_mgr.state.initial_balance == 0.0


@pytest.mark.asyncio
async def test_restore_current_session_missing_state_mgr():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_store = AsyncMock()
    del orch.state_mgr
    await restore_current_session_targets(orch)


@pytest.mark.asyncio
async def test_clear_current_session_without_store():
    orch = MagicMock()
    orch.state_store = None
    await clear_current_session_redis_keys(orch)


def test_current_session_redis_payload_rejects_zero_target():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr = StateManager()
    orch.state_mgr.state.initial_balance = 1000.0
    orch.state_mgr.state.daily_stop_win_target = 0.0
    assert current_session_redis_payload(orch) == (None, None)


@pytest.mark.asyncio
async def test_bootstrap_without_state_store():
    orch = MagicMock()
    orch._session_targets_bootstrapped = False
    orch.config = {"risk_management": {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.01}}}
    orch.state_mgr = StateManager()
    orch.risk_manager = RiskManager(orch.config["risk_management"])
    orch.state_store = None
    orch.logger = MagicMock()
    await bootstrap_active_session_targets(orch, 1000.0)
    assert orch._session_targets_bootstrapped is True


def test_current_session_redis_payload_rejects_invalid_values():
    orch = MagicMock()
    orch._session_targets_bootstrapped = True
    orch.state_mgr = MagicMock()
    orch.state_mgr.state = MagicMock()
    orch.state_mgr.state.initial_balance = "x"
    orch.state_mgr.state.daily_stop_win_target = 10.0
    assert current_session_redis_payload(orch) == (None, None)
