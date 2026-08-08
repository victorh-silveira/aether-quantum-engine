from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models.trade import TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import (
    TradeHandler,
    _contract_duration_seconds,
    build_proposal_request,
    resolve_api_contract_type,
)


@pytest.fixture
def mock_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def trade_handler(mock_ws):
    config = {"risk_management": {"params": {"duration": 2, "duration_unit": "m"}}}
    mock_ws.request_timeout = 30
    return TradeHandler(mock_ws, config)


@pytest.mark.asyncio
async def test_trade_handler_buy_via_bulk_purchase_rest(mock_ws):
    mock_ws.request_timeout = 30
    auth = MagicMock()
    auth.mode = "demo"
    auth.get_pat.return_value = "pat_x"
    auth.account_id_override = None
    auth.rest_client.return_value.bulk_purchase = AsyncMock(
        return_value={
            "buy_price": "0.35",
            "contract_id": "77",
            "payout": "0.66",
            "purchase_time": 1000,
            "start_time": 1000,
            "shortcode": "CALL_R_10_0.66_1000_1120_S0P_0",
            "transaction_id": "55",
        }
    )
    handler = TradeHandler(mock_ws, {"risk_management": {"params": {"duration": 2, "duration_unit": "m"}}}, auth=auth)
    handler.trading_transport = "rest"
    handler.deriv_account_id = "DOT1"
    contract = await handler.buy_with_parameters("R_10", TradeDirection.CALL, 0.35)
    assert contract.contract_id == 77
    assert contract.expiry_time == 1120
    assert contract.payout == pytest.approx(0.66)
    mock_ws.send.assert_not_called()


@pytest.mark.asyncio
async def test_trade_handler_bulk_purchase_requires_auth_pat_account(mock_ws):
    mock_ws.request_timeout = 30
    handler = TradeHandler(mock_ws, {"risk_management": {"params": {}}}, auth=None)
    handler.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="AuthManager"):
        await handler.buy_with_parameters("R_10", TradeDirection.CALL, 1.0)
    auth = MagicMock()
    auth.get_pat.return_value = None
    auth.account_id_override = None
    handler = TradeHandler(mock_ws, {"risk_management": {"params": {}}}, auth=auth)
    handler.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="AETHER_DERIV_PAT"):
        await handler.buy_with_parameters("R_10", TradeDirection.CALL, 1.0)
    auth.get_pat.return_value = "pat"
    handler.deriv_account_id = ""
    with pytest.raises(RuntimeError, match="deriv_account_id"):
        await handler.buy_with_parameters("R_10", TradeDirection.CALL, 1.0)


@pytest.mark.asyncio
async def test_trade_handler_schedule_profit_table_audit(trade_handler):
    orch = MagicMock()
    with patch(
        "src.infrastructure.handlers.trade_handler.schedule_profit_table_audit",
    ) as schedule:
        trade_handler.schedule_profit_table_audit(orch, reason="test")
        schedule.assert_called_once_with(orch, reason="test")


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_success(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {
            "proposal": {
                "id": "prop-abc",
                "ask_price": 10.0,
                "payout": 18.5,
                "date_expiry": 1900000000,
                "longcode": "Win contract",
            }
        },
        {"buy": {"contract_id": 999, "buy_price": 10.0, "payout": 18.5, "longcode": "Win contract"}},
    ]

    contract = await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)
    assert contract.contract_id == 999
    assert contract.proposal_id == "prop-abc"
    assert contract.status == TradeStatus.OPEN
    assert contract.symbol == "R_10"
    assert contract.direction == TradeDirection.CALL
    assert contract.stake == 10.0

    proposal_req = mock_ws.send.call_args_list[0].args[0]
    buy_req = mock_ws.send.call_args_list[1].args[0]
    assert proposal_req["underlying_symbol"] == "R_10"
    assert proposal_req["contract_type"] == "CALL"
    assert "subscribe" not in proposal_req
    assert buy_req["buy"] == "prop-abc"
    assert buy_req["price"] == 10.0


@pytest.mark.asyncio
async def test_trade_handler_buy_uses_date_expiry_from_api(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {"proposal": {"id": "p1", "ask_price": 2.34, "date_expiry": 1900000000}},
        {"buy": {"contract_id": 1001, "buy_price": 2.34, "payout": 4.26}},
    ]
    contract = await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 2.34)
    assert contract.expiry_time == 1900000000


@pytest.mark.asyncio
async def test_trade_handler_buy_proposal_invalid_payload(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": "bad"}
    with pytest.raises(RuntimeError, match="resposta sem proposal"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_proposal_missing_id(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": {"ask_price": 10.0}}
    with pytest.raises(RuntimeError, match="id ausente"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_proposal_error(trade_handler, mock_ws):
    mock_ws.send.return_value = {"error": {"message": "Invalid symbol"}}
    with pytest.raises(RuntimeError, match="Erro na proposta: Invalid symbol"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_error(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {"proposal": {"id": "p1", "ask_price": 10.0}},
        {"error": {"message": "Insufficient balance"}},
    ]
    with pytest.raises(RuntimeError, match="Erro na compra direta: Insufficient balance"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_trade_handler_buy_with_parameters_multiplier(trade_handler, mock_ws):
    params = {
        "contract_type": "MULTIPLIER",
        "multiplier": 100,
        "cancellation": "1h",
        "limit_order": {"take_profit": 10.0},
        "barrier": "+0.1",
    }
    mock_ws.send.side_effect = [
        {"proposal": {"id": "m1", "ask_price": 10.0}},
        {"buy": {"contract_id": 999, "buy_price": 10.0, "payout": 0.0, "longcode": "Multiplier contract"}},
    ]

    contract = await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0, params=params)
    assert contract.contract_id == 999

    proposal_req = mock_ws.send.call_args_list[0].args[0]
    assert proposal_req["contract_type"] == "MULTUP"
    assert proposal_req["multiplier"] == 100
    assert proposal_req["barrier"] == "+0.1"


def test_build_proposal_request_rise_fall():
    req = build_proposal_request("R_10", TradeDirection.PUT, 5.0, {"duration": 1, "duration_unit": "m"})
    assert req["underlying_symbol"] == "R_10"
    assert req["contract_type"] == "PUT"
    assert req["duration"] == 1


def test_resolve_api_contract_type_multiplier():
    assert resolve_api_contract_type(TradeDirection.CALL, {"contract_type": "MULTIPLIER"}) == "MULTUP"
    assert resolve_api_contract_type(TradeDirection.PUT, {"contract_type": "MULTIPLIER"}) == "MULTDOWN"


def test_contract_duration_seconds_units():
    assert _contract_duration_seconds({"duration": 10, "duration_unit": "s"}) == 10
    assert _contract_duration_seconds({"duration": 10, "duration_unit": "t"}) == 20
    assert _contract_duration_seconds({"duration": 1, "duration_unit": "d"}) == 86400
    assert _contract_duration_seconds({"duration": 5, "duration_unit": "invalid"}) == 300
