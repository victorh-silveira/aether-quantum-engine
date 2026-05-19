from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models.trade import TradeDirection, TradeStatus
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
async def test_trade_handler_buy_with_parameters_success(trade_handler, mock_ws):
    mock_ws.send.return_value = {
        "buy": {"contract_id": 999, "buy_price": 10.0, "payout": 18.5, "longcode": "Win contract"}
    }

    contract = await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0)
    assert contract.contract_id == 999
    assert contract.status == TradeStatus.OPEN
    assert contract.symbol == "1HZ75V"
    assert contract.direction == TradeDirection.CALL
    assert contract.stake == 10.0


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_error(trade_handler, mock_ws):
    mock_ws.send.return_value = {"error": {"message": "Insufficient balance"}}
    with pytest.raises(RuntimeError, match="Erro na compra direta: Insufficient balance"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_multiplier(trade_handler, mock_ws):
    params = {
        "contract_type": "MULTIPLIER",
        "multiplier": 100,
        "cancellation": "1h",
        "limit_order": {"take_profit": 10.0},
        "barrier": "+0.1",
    }
    mock_ws.send.return_value = {
        "buy": {"contract_id": 999, "buy_price": 10.0, "payout": 0.0, "longcode": "Multiplier contract"}
    }

    contract = await trade_handler.buy_with_parameters("frxEURUSD", TradeDirection.CALL, 10.0, params=params)
    assert contract.contract_id == 999

    args, _ = mock_ws.send.call_args
    assert args[0]["parameters"]["contract_type"] == "MULTUP"
    assert args[0]["parameters"]["multiplier"] == 100
    assert args[0]["parameters"]["barrier"] == "+0.1"
