from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_orders import place_order
from src.domain.models.trade import Contract, TradeDirection, TradeStatus


@pytest.mark.asyncio
async def test_place_order_rest_transport_schedules_settlement():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.trading_transport = "rest"
    orch.deriv_account_id = "DOT1"
    orch.state.balance = 100.0
    orch.config = {
        "risk_management": {"params": {"stake_min": 0.35, "duration": 2, "duration_unit": "m"}},
        "orchestrator": {"execution": {"settlement_request_timeout_seconds": 5.0}},
    }
    orch.risk_manager.contract_to_symbol = {}
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(
        return_value=[MagicMock(account_id="DOT1", balance=99.65)]
    )
    contract = Contract(
        contract_id=5,
        proposal_id="t",
        status=TradeStatus.OPEN,
        buy_price=0.35,
        payout=0.66,
        symbol="R_10",
        direction=TradeDirection.CALL,
        stake=0.35,
        expiry_time=10,
        longcode="x",
    )
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=contract)
    executor = MagicMock()
    executor.orch = orch
    with patch("src.application.services.orchestrator.execution_orders.schedule_rest_contract_settlement") as sched:
        out = await place_order(executor, "R_10", TradeDirection.CALL, 0.35)
    assert out is contract
    sched.assert_called_once()
    assert orch.state.balance == 99.65


@pytest.mark.asyncio
async def test_place_order_rest_transport_refresh_balance_failure():
    orch = MagicMock()
    orch._active_cycle_id = 2
    orch.trading_transport = "rest"
    orch.deriv_account_id = "DOT1"
    orch.state.balance = 50.0
    orch.config = {
        "risk_management": {"params": {"stake_min": 0.35}},
        "orchestrator": {"execution": {}},
    }
    orch.risk_manager.contract_to_symbol = {}
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(side_effect=RuntimeError("x"))
    contract = Contract(
        contract_id=6,
        proposal_id="t",
        status=TradeStatus.OPEN,
        buy_price=0.35,
        payout=0.66,
        symbol="R_10",
        direction=TradeDirection.PUT,
        stake=0.35,
        expiry_time=10,
        longcode="x",
    )
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=contract)
    executor = MagicMock()
    executor.orch = orch
    with patch("src.application.services.orchestrator.execution_orders.schedule_rest_contract_settlement"):
        out = await place_order(executor, "R_10", TradeDirection.PUT, 0.35)
    assert out.contract_id == 6
