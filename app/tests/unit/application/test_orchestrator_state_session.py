"""Cobertura de restore de sessao Redis."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.orchestrator_state_session import (
    restore_market_signatures,
    restore_session_hash,
)


@pytest.mark.asyncio
async def test_restore_session_hash_all_fields():
    orch = MagicMock()
    store = AsyncMock()
    store.get_hash.return_value = {
        "initial_balance": "10",
        "current_balance": "12",
        "daily_stop_win_target": "5",
        "total_trades_today": "2",
        "stop_win_triggered": "true",
        "day_key": "3",
    }
    await restore_session_hash(orch, store)
    mgr = orch.state_mgr.state
    assert mgr.initial_balance == 10.0
    assert mgr.stop_win_triggered is True


@pytest.mark.asyncio
async def test_restore_market_signatures_no_anchor():
    orch = MagicMock()
    orch.anchor = None
    store = AsyncMock()
    store.get_string.return_value = None
    await restore_market_signatures(orch, store)
