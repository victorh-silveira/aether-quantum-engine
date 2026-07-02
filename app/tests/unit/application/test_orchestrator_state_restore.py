"""Testes de restore de estado do orquestrador."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.orchestrator_state_restore import (
    bar_epoch_already_processed,
    mark_bar_processed,
    restore_orchestrator_state,
)
from src.domain.risk.risk_manager import RiskManager


@pytest.mark.asyncio
async def test_restore_orchestrator_state():
    orch = MagicMock()
    orch.state_store = AsyncMock()
    orch.state_store.load_snapshot.return_value = {
        "risk": {"consecutive_losses_linear": 3, "pending_loss": {"RDBEAR": 2.0}},
        "total_session_profit": 5.0,
    }

    async def _get_hash(key: str):
        if key == "state:pending_loss":
            return {"RDBEAR": "2.0"}
        if key == "session:daily":
            return {"initial_balance": "100.0", "day_key": "1"}
        return {}

    async def _get_string(key: str):
        if key == "market_sig":
            return "sig"
        if key == "bar_sig:RDBEAR":
            return "99"
        return None

    orch.state_store.get_hash.side_effect = _get_hash
    orch.state_store.get_string.side_effect = _get_string
    orch.config = {"risk_management": {"params": {"compounding_enabled": False}}}
    orch.risk_manager = RiskManager({"params": {}, "kelly": {}, "limits": {}})
    orch.anchor = "RDBEAR"
    await restore_orchestrator_state(orch)
    assert orch.risk_manager.consecutive_losses_linear == 3
    assert orch.risk_manager.pending_loss["RDBEAR"] == 2.0
    assert orch.last_data_signature == "sig"


@pytest.mark.asyncio
async def test_bar_sig_helpers():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.state_store = AsyncMock()
    orch.state_store.get_string.return_value = "42"
    assert await bar_epoch_already_processed(orch, "RDBEAR", 42) is True
    await mark_bar_processed(orch, "RDBEAR", 43)
    orch.state_store.set_string.assert_called()
