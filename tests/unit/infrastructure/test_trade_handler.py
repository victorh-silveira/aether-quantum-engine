from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models.trade import Proposal, TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import TradeHandler


@pytest.fixture
def mock_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def trade_handler(mock_ws):
    config = {"risk_management": {"params": {"duration": 15, "duration_unit": "m"}}}
    return TradeHandler(mock_ws, config)


@pytest.mark.asyncio
async def test_trade_handler_get_proposal_success(trade_handler, mock_ws):
    mock_ws.send.return_value = {
        "proposal": {"id": "prop123", "payout": 18.5, "spot": 1.1000, "expiry_time": 1700000000}
    }

    proposal = await trade_handler.get_proposal("1HZ75V", TradeDirection.CALL, 10.0)
    assert proposal.proposal_id == "prop123"
    assert proposal.payout == 18.5


@pytest.mark.asyncio
async def test_trade_handler_get_proposal_error(trade_handler, mock_ws):
    mock_ws.send.return_value = {"error": {"message": "Invalid stake"}}
    with pytest.raises(RuntimeError, match="Erro na proposta: Invalid stake"):
        await trade_handler.get_proposal("1HZ75V", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_contract_success(trade_handler, mock_ws):
    proposal = Proposal(
        proposal_id="prop123",
        symbol="1HZ75V",
        direction=TradeDirection.CALL,
        stake=10.0,
        payout=18.5,
        spot=1.1000,
        expiry_time=1700000000,
    )
    mock_ws.send.return_value = {
        "buy": {"contract_id": 999, "buy_price": 10.0, "payout": 18.5, "longcode": "Win contract"}
    }

    contract = await trade_handler.buy_contract(proposal)
    assert contract.contract_id == 999
    assert contract.status == TradeStatus.OPEN
    assert contract.symbol == "1HZ75V"
    assert contract.direction == TradeDirection.CALL
    assert contract.stake == 10.0


@pytest.mark.asyncio
async def test_trade_handler_buy_contract_error(trade_handler, mock_ws):
    proposal = Proposal(
        proposal_id="prop123",
        symbol="1HZ75V",
        direction=TradeDirection.CALL,
        stake=10.0,
        payout=18.5,
        spot=1.1000,
        expiry_time=1700000000,
    )
    mock_ws.send.return_value = {"error": {"message": "Insufficient balance"}}
    with pytest.raises(RuntimeError, match="Erro na compra: Insufficient balance"):
        await trade_handler.buy_contract(proposal)


@pytest.mark.asyncio
async def test_trade_handler_get_proposal_with_barrier(trade_handler, mock_ws):
    trade_handler.config["risk_management"]["params"]["barrier"] = "+0.1"
    mock_ws.send.return_value = {
        "proposal": {"id": "prop123", "payout": 18.5, "spot": 1.1000, "expiry_time": 1700000000}
    }
    await trade_handler.get_proposal("1HZ75V", TradeDirection.CALL, 10.0)
    args, _ = mock_ws.send.call_args
    assert args[0]["barrier"] == "+0.1"


@pytest.mark.asyncio
async def test_trade_handler_get_proposal_multiplier(trade_handler, mock_ws):
    params = {
        "contract_type": "MULTIPLIER",
        "multiplier": 100,
        "cancellation": "1h",
        "limit_order": {"take_profit": 10.0},
    }
    mock_ws.send.return_value = {
        "proposal": {"id": "prop123", "payout": 18.5, "spot": 1.1000, "date_expiry": 1700000000}
    }

    proposal = await trade_handler.get_proposal("frxEURUSD", TradeDirection.CALL, 10.0, params=params)
    assert proposal.proposal_id == "prop123"

    args, _ = mock_ws.send.call_args
    assert args[0]["contract_type"] == "MULTUP"
    assert args[0]["multiplier"] == 100
    assert args[0]["cancellation"] == "1h"
    assert args[0]["symbol"] == "frxEURUSD"
    assert "limit_order" in args[0]
    assert args[0]["limit_order"]["take_profit"] == 10.0
    assert "duration" not in args[0]
