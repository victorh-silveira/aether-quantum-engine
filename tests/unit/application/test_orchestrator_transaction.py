from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_on_transaction_reconciles_tracked_contract(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.active_contracts[99] = MagicMock()
        with patch(
            "src.application.services.orchestrator.reconcile_single_contract",
            AsyncMock(),
        ) as mock_rec:
            await orch._on_transaction({"transaction": {"contract_id": 99}})
        mock_rec.assert_awaited_once_with(orch, 99)


@pytest.mark.asyncio
async def test_on_transaction_ignores_unknown_contract(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch(
            "src.application.services.orchestrator.reconcile_single_contract",
            AsyncMock(),
        ) as mock_rec:
            await orch._on_transaction({"transaction": {"contract_id": 999}})
        mock_rec.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscribe_account_transactions_failure(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        mock_ws.send = AsyncMock(side_effect=RuntimeError("tx"))
        orch = Orchestrator(orch_config, "token")
        await orch._subscribe_account_transactions()


@pytest.mark.asyncio
async def test_on_transaction_invalid_payload(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        await orch._on_transaction({})
        await orch._on_transaction({"transaction": "x"})
        await orch._on_transaction({"transaction": {}})
