from unittest.mock import MagicMock

import pytest

from src.application.services.orchestrator.trading_cycle_entry import (
    acquire_trading_cycle_lock,
    trading_cycle_entry_allowed,
)


def test_trading_cycle_entry_blocked_when_is_trading(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    assert trading_cycle_entry_allowed(orch) is False


def test_trading_cycle_entry_waits_for_settlement(orch_ready):
    orch = orch_ready
    orch.state.active_contracts = {1: object()}
    orch.logger = MagicMock()
    assert trading_cycle_entry_allowed(orch) is False
    orch.logger.info.assert_called_once()
    assert orch._settlement_wait_logged is True


def test_trading_cycle_entry_allowed(orch_ready):
    orch = orch_ready
    orch._settlement_wait_logged = True
    assert trading_cycle_entry_allowed(orch) is True
    assert orch._settlement_wait_logged is False


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
