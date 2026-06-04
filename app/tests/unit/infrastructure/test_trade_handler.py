from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models.trade import TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import TradeHandler, _contract_duration_seconds


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

    contract = await trade_handler.buy_with_parameters("RDBULL", TradeDirection.CALL, 10.0)
    assert contract.contract_id == 999
    assert contract.status == TradeStatus.OPEN
    assert contract.symbol == "RDBULL"
    assert contract.direction == TradeDirection.CALL
    assert contract.stake == 10.0


@pytest.mark.asyncio
async def test_trade_handler_buy_uses_date_expiry_from_api(trade_handler, mock_ws):
    mock_ws.send.return_value = {
        "buy": {
            "contract_id": 1001,
            "buy_price": 2.34,
            "payout": 4.26,
            "date_expiry": 1900000000,
        }
    }
    contract = await trade_handler.buy_with_parameters("RDBEAR", TradeDirection.CALL, 2.34)
    assert contract.expiry_time == 1900000000


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_error(trade_handler, mock_ws):
    mock_ws.send.return_value = {"error": {"message": "Insufficient balance"}}
    with pytest.raises(RuntimeError, match="Erro na compra direta: Insufficient balance"):
        await trade_handler.buy_with_parameters("RDBULL", TradeDirection.CALL, 10.0)


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

    contract = await trade_handler.buy_with_parameters("RDBULL", TradeDirection.CALL, 10.0, params=params)
    assert contract.contract_id == 999

    args, _ = mock_ws.send.call_args
    assert args[0]["parameters"]["contract_type"] == "MULTUP"
    assert args[0]["parameters"]["multiplier"] == 100
    assert args[0]["parameters"]["barrier"] == "+0.1"


def test_contract_duration_seconds_units():
    assert _contract_duration_seconds({"duration": 10, "duration_unit": "s"}) == 10
    assert _contract_duration_seconds({"duration": 10, "duration_unit": "t"}) == 20
    assert _contract_duration_seconds({"duration": 1, "duration_unit": "d"}) == 86400
    assert _contract_duration_seconds({"duration": 5, "duration_unit": "invalid"}) == 300
