from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_contract_update_buffers_result_logs_during_dispatch(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._buffer_result_logs = True
        orch.state.active_contracts[1] = Contract(
            contract_id=1,
            proposal_id="p1",
            status=TradeStatus.OPEN,
            buy_price=10.0,
            payout=18.0,
            symbol="R_10",
            direction=TradeDirection.PUT,
            stake=10.0,
            expiry_time=0,
        )
        orch.risk_manager.contract_to_symbol[1] = "R_10"
        orch.risk_manager.active_contract_ids = [1]
        data = {
            "proposal_open_contract": {
                "contract_id": 1,
                "is_settled": 1,
                "status": "won",
                "profit": 10.0,
                "balance_after": 1010.0,
            }
        }
        with patch.object(orch.logger, "info") as mock_info:
            await orch._on_contract_update(data)
        assert mock_info.call_count == 0
        assert len(orch._pending_result_logs) == 1
        joined = "\n".join(orch._pending_result_logs)
        assert "[C0000] RESOLVED || STATUS: WIN" in joined
        assert "P&L:" in joined
        assert "+10.00" in joined


@pytest.mark.asyncio
async def test_contract_update_increments_loss_counter(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._buffer_result_logs = True
        orch.state.active_contracts[2] = Contract(
            contract_id=2,
            proposal_id="p2",
            status=TradeStatus.OPEN,
            buy_price=10.0,
            payout=18.0,
            symbol="R_10",
            direction=TradeDirection.CALL,
            stake=10.0,
            expiry_time=0,
        )
        orch.risk_manager.contract_to_symbol[2] = "R_10"
        data = {
            "proposal_open_contract": {
                "contract_id": 2,
                "is_settled": 1,
                "status": "lost",
                "profit": -10.0,
                "balance_after": 990.0,
            }
        }
        await orch._on_contract_update(data)
        assert orch._session_losses == 1
