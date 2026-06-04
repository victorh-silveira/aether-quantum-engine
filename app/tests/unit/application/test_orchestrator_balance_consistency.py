from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_contract_update_ignores_inconsistent_balance_after(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.balance = 9940.0
        orch.state.active_contracts[1] = Contract(
            contract_id=1,
            proposal_id="p1",
            status=TradeStatus.OPEN,
            buy_price=10.0,
            payout=18.0,
            symbol="R_50",
            direction=TradeDirection.PUT,
            stake=10.0,
            expiry_time=0,
        )
        orch.risk_manager.contract_to_symbol[1] = "R_50"
        data = {
            "proposal_open_contract": {
                "contract_id": 1,
                "is_settled": 1,
                "status": "won",
                "profit": 114.12,
                "balance_after": 9930.31,
            }
        }
        await orch._on_contract_update(data)
        assert orch.state.balance == pytest.approx(10054.12, abs=1e-6)
