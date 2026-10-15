from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models.trade import TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import (
    TradeHandler,
    _contract_duration_seconds,
    build_proposal_request,
    resolve_api_contract_type,
)
from src.infrastructure.handlers.trade_handler_errors import PurchaseOutcomeUnknownError


@pytest.mark.asyncio
async def test_quote_crossing_m5_boundary_cannot_buy(trade_handler, mock_ws, monkeypatch):
    from src.infrastructure.handlers import trade_handler_requests

    monkeypatch.setattr(trade_handler_requests.time, "time", lambda: 900)
    mock_ws.send.return_value = {"proposal": {"id": "quote", "ask_price": 10.0, "payout": 18.0}}
    with pytest.raises(RuntimeError, match="Sinal expirou"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0, params={"_signal_candle_epoch": 300})
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposal", "expected"),
    [
        ({"id": "quote", "ask_price": 51.0, "payout": 91.8}, "excede o teto"),
        ({"id": "quote", "ask_price": 40.0, "payout": 40.0}, "sem payout valido"),
        ({"id": "quote", "ask_price": 40.0, "payout": 71.2}, "Payout final"),
    ],
)
async def test_pullback_revalidates_cap_and_quoted_rate(trade_handler, mock_ws, proposal, expected):
    mock_ws.send.return_value = {"proposal": proposal}
    with pytest.raises(RuntimeError, match=expected):
        await trade_handler.buy_with_parameters(
            "R_10",
            TradeDirection.CALL,
            40.0,
            params={"_pullback_stake_cap": 50.0, "_pullback_quote_rate": 0.8},
        )
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_signal_period_cannot_buy(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": {"id": "quote", "ask_price": 10.0, "payout": 18.0}}
    with pytest.raises(RuntimeError, match="Periodo do sinal invalido"):
        await trade_handler.buy_with_parameters(
            "R_10",
            TradeDirection.CALL,
            10.0,
            params={"_signal_candle_epoch": 300, "_indicator_period_seconds": 600},
        )
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("period", [900, 3600])
async def test_period_signal_cannot_buy_after_entry_window(trade_handler, mock_ws, monkeypatch, period):
    from src.infrastructure.handlers import trade_handler_requests

    monkeypatch.setattr(trade_handler_requests.time, "time", lambda: period * 3 + 31)
    mock_ws.send.return_value = {"proposal": {"id": "quote", "ask_price": 10.0, "payout": 18.0}}
    with pytest.raises(RuntimeError, match="Sinal expirou"):
        await trade_handler.buy_with_parameters(
            "R_10",
            TradeDirection.CALL,
            10.0,
            params={"_signal_candle_epoch": period * 2, "_indicator_period_seconds": period},
        )
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_buy_timeout_is_reported_as_unknown_outcome(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote", "ask_price": 10.0, "payout": 18.0}},
        TimeoutError("no ACK"),
    ]
    with pytest.raises(PurchaseOutcomeUnknownError, match="resultado no broker desconhecido"):
        await trade_handler.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)
    assert mock_ws.send.await_count == 2


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
async def test_trade_handler_rest_rejects_guarded_purchase_before_bulk_call(mock_ws):
    auth = MagicMock()
    handler = TradeHandler(mock_ws, {"risk_management": {"params": {}}}, auth=auth)
    handler.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="REST sem cotacao final"):
        await handler.buy_with_parameters(
            "1HZ75V", TradeDirection.PUT, 10.0, params={"_quote_guard_side_probability": 0.65}
        )
    auth.rest_client.assert_not_called()


@pytest.mark.asyncio
async def test_trade_handler_rest_rejects_quote_guard_from_config_without_private_params(mock_ws):
    auth = MagicMock()
    handler = TradeHandler(
        mock_ws,
        {"orchestrator": {"execution": {"require_quote_edge": True}}, "risk_management": {"params": {}}},
        auth=auth,
    )
    handler.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="REST sem cotacao final"):
        await handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0)
    auth.rest_client.assert_not_called()


@pytest.mark.asyncio
async def test_trade_handler_ws_requires_valid_quote_guard_before_proposal(mock_ws):
    handler = TradeHandler(
        mock_ws,
        {"orchestrator": {"execution": {"require_quote_edge": True}}, "risk_management": {"params": {}}},
    )
    for params in ({}, {"_quote_guard_side_probability": "nan"}, {"_quote_guard_side_probability": "bad"}):
        with pytest.raises(RuntimeError, match="sem probabilidade valida"):
            await handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
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
    assert contract.quote_price == 10.0
    assert contract.quote_payout == 18.5
    assert contract.quote_ev is None

    proposal_req = mock_ws.send.call_args_list[0].args[0]
    buy_req = mock_ws.send.call_args_list[1].args[0]
    assert proposal_req["underlying_symbol"] == "R_10"
    assert proposal_req["contract_type"] == "CALL"
    assert "subscribe" not in proposal_req
    assert buy_req["buy"] == "prop-abc"
    assert buy_req["price"] == 10.0


@pytest.mark.asyncio
async def test_trade_handler_purchase_audit_does_not_retry_confirmed_buy(mock_ws, caplog):
    mock_ws.request_timeout = 30
    mock_ws.send.side_effect = [
        {"proposal": {"id": "p", "ask_price": 10}},
        {"buy": {"contract_id": 9, "transaction_id": 11, "start_time": 100, "buy_price": 10, "payout": 19}},
    ]
    writer = MagicMock()
    writer.enqueue_contract_audit = AsyncMock()
    handler = TradeHandler(
        mock_ws,
        {"trading": {"mode": "real"}, "risk_management": {"params": {}}},
        market_writer=writer,
    )
    contract = await handler.buy_with_parameters("1HZ75V", TradeDirection.PUT, 10)
    assert contract.contract_id == 9
    row = writer.enqueue_contract_audit.call_args.args[0]
    assert row["transaction_buy_id"] == "11"
    assert row["account_mode"] == "real"
    assert row["request_epoch_ms"] <= row["ack_epoch_ms"]
    writer.enqueue_contract_audit.side_effect = RuntimeError("db down")
    await handler._record_purchase_audit({"contract_id": 9}, "1HZ75V", TradeDirection.PUT, 1, 2)
    assert "sem captura" in caplog.text


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


@pytest.mark.asyncio
async def test_trade_handler_fetch_proposal_payout_success(trade_handler, mock_ws):
    mock_ws.send.return_value = {
        "proposal": {
            "id": "prop_test",
            "ask_price": "10.00",
            "payout": "18.50",
        }
    }
    rate = await trade_handler.fetch_proposal_payout("1HZ75V", TradeDirection.CALL, 10.0)
    assert rate == pytest.approx(0.85)
    assert trade_handler.latest_payout_rate == pytest.approx(0.85)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [5, 15, 60])
async def test_proposal_snapshot_uses_period_contract_duration(trade_handler, mock_ws, duration):
    mock_ws.send.return_value = {"proposal": {"id": "quote-v1", "ask_price": "10.00", "payout": "18.50"}}
    quote = await trade_handler.fetch_proposal_snapshot(
        "R_25",
        TradeDirection.PUT,
        10.0,
        params={"duration": duration, "duration_unit": "m"},
    )
    assert quote["proposal_id"] == "quote-v1"
    assert quote["price"] == 10.0
    assert quote["payout"] == 18.5
    assert quote["rate"] == pytest.approx(0.85)
    assert mock_ws.send.await_args.args[0]["duration"] == duration


@pytest.mark.asyncio
@pytest.mark.parametrize("payout", ["18.00", "0.00"])
async def test_final_buy_proposal_blocks_lost_quote_edge(trade_handler, mock_ws, payout):
    mock_ws.send.return_value = {"proposal": {"id": "new_quote", "ask_price": "10.00", "payout": payout}}
    params = {
        "duration": 5,
        "duration_unit": "m",
        "_quote_guard_side_probability": 0.55,
        "_quote_guard_min_edge": 0.01,
    }
    with pytest.raises(RuntimeError, match="Cotacao final Rise/Fall perdeu vantagem") as caught:
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
    if payout == "18.00":
        assert "p_minima=0.5611" in str(caught.value)
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_buy_proposal_blocks_degraded_payout_rate(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": {"id": "new_quote", "ask_price": "10.00", "payout": "17.00"}}
    params = {"duration": 5, "duration_unit": "m", "min_payout_rate": 0.76}
    with pytest.raises(RuntimeError, match="Cotacao final Rise/Fall perdeu vantagem"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_buy_accepts_positive_ev_below_old_fixed_payout_floor(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote_783", "ask_price": "10.00", "payout": "17.83"}},
        {"buy": {"contract_id": 783, "buy_price": "10.00", "payout": "17.83"}},
    ]
    params = {
        "duration": 5,
        "duration_unit": "m",
        "min_payout_rate": 0.0,
        "_quote_guard_side_probability": 0.6752,
        "_quote_guard_min_edge": 0.0,
        "_quote_guard_safety_margin": 0.01,
    }
    contract = await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.PUT, 10.0, params=params)
    assert contract.contract_id == 783
    assert contract.quote_ev == pytest.approx(0.6752 * 1.783 - 1.0)
    assert trade_handler.latest_payout_rate == pytest.approx(0.783)
    assert mock_ws.send.await_count == 2


@pytest.mark.asyncio
async def test_final_buy_accepts_negative_ev_only_for_demo_pilot(trade_handler, mock_ws):
    trade_handler.config["trading"] = {"mode": "demo"}
    mock_ws.send.side_effect = [
        {"proposal": {"id": "indicator_quote", "ask_price": "10.00", "payout": "17.81"}},
        {"buy": {"contract_id": 784, "buy_price": "10.00", "payout": "17.81"}},
    ]
    params = {
        "duration": 5,
        "duration_unit": "m",
        "min_payout_rate": 0.0,
        "_quote_guard_side_probability": 0.528,
        "_quote_guard_min_edge": -1.0,
        "_quote_guard_safety_margin": 0.0,
        "_indicator_demo_pilot": True,
    }
    contract = await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 10.0, params=params)
    assert contract.contract_id == 784
    assert contract.quote_ev == pytest.approx(0.528 * 1.781 - 1.0)
    assert mock_ws.send.await_count == 2


@pytest.mark.asyncio
async def test_pullback_demo_pilot_uses_valid_quote_without_probability(trade_handler, mock_ws):
    trade_handler.config["trading"] = {"mode": "demo"}
    trade_handler.market_writer = MagicMock(record_quote_observation=AsyncMock())
    mock_ws.send.side_effect = [
        {"proposal": {"id": "setup_quote", "ask_price": "1.00", "payout": "1.78"}},
        {"buy": {"contract_id": 785, "buy_price": "1.00", "payout": "1.78"}},
    ]
    params = {
        "duration": 5,
        "duration_unit": "m",
        "_indicator_model_version": "pullback-1-test",
        "_indicator_demo_pilot": True,
    }
    contract = await trade_handler.buy_with_parameters("R_25", TradeDirection.CALL, 1.0, params=params)
    assert contract.quote_ev is None
    assert contract.quote_payout == 1.78
    assert mock_ws.send.await_count == 2


@pytest.mark.asyncio
async def test_pullback_pilot_bypass_rejected_in_real_and_without_payout(trade_handler, mock_ws):
    params = {"_indicator_model_version": "pullback-1-test", "_indicator_demo_pilot": True}
    trade_handler.market_writer = MagicMock(record_quote_observation=AsyncMock())
    trade_handler.config["trading"] = {"mode": "real"}
    with pytest.raises(RuntimeError, match="somente em DEMO"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.CALL, 1.0, params=params)
    mock_ws.send.assert_not_awaited()
    trade_handler.config["trading"] = {"mode": "demo"}
    mock_ws.send.return_value = {"proposal": {"id": "sem_payout", "ask_price": "1.00"}}
    with pytest.raises(RuntimeError, match="sem payout valido"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.CALL, 1.0, params=params)


@pytest.mark.asyncio
async def test_negative_ev_cannot_be_enabled_in_real_even_with_pilot_flag(trade_handler, mock_ws):
    trade_handler.config["trading"] = {"mode": "real"}
    params = {
        "_quote_guard_side_probability": 0.528,
        "_quote_guard_min_edge": -1.0,
        "_indicator_demo_pilot": True,
    }
    with pytest.raises(RuntimeError, match="somente no piloto DEMO"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 10.0, params=params)
    mock_ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_proposal_uses_one_direct_buy_for_current_indicator_candle(trade_handler, mock_ws, monkeypatch):
    from src.infrastructure.handlers import trade_handler_buy_recovery

    trade_handler.config["trading"] = {"mode": "demo"}
    monkeypatch.setattr(trade_handler_buy_recovery.time, "time", lambda: 620)
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote_r25", "ask_price": "59.86", "payout": "106.60"}},
        {"error": {"message": "Unknown contract proposal"}},
        {"buy": {"contract_id": 785, "buy_price": "59.86", "payout": "106.60"}},
    ]
    params = {
        "duration": 5,
        "duration_unit": "m",
        "_quote_guard_side_probability": 0.528,
        "_quote_guard_min_edge": -1.0,
        "_indicator_demo_pilot": True,
        "_indicator_direct_buy_fallback": True,
        "_indicator_candle_epoch": 300,
    }
    contract = await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 59.86, params=params)
    assert contract.contract_id == 785
    assert contract.proposal_id == "direct_parameters"
    assert contract.quote_price is None
    assert contract.quote_payout is None
    assert contract.quote_ev is None
    direct = mock_ws.send.await_args_list[2].args[0]
    assert direct == {
        "buy": "1",
        "price": 59.86,
        "parameters": {
            "amount": 59.86,
            "basis": "stake",
            "contract_type": "PUT",
            "currency": "USD",
            "underlying_symbol": "R_25",
            "duration": 5,
            "duration_unit": "m",
        },
    }


@pytest.mark.asyncio
async def test_unknown_proposal_does_not_retry_after_candle_expires(trade_handler, mock_ws, monkeypatch):
    from src.infrastructure.handlers import trade_handler_buy_recovery

    monkeypatch.setattr(trade_handler_buy_recovery.time, "time", lambda: 900)
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote_r25", "ask_price": "10.00", "payout": "17.81"}},
        {"error": {"message": "Unknown contract proposal"}},
    ]
    params = {"_indicator_direct_buy_fallback": True, "_indicator_candle_epoch": 300}
    with pytest.raises(RuntimeError, match="sinal expirou"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 10.0, params=params)
    assert mock_ws.send.await_count == 2


@pytest.mark.asyncio
async def test_unknown_proposal_without_indicator_policy_does_not_retry(trade_handler, mock_ws):
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote_r25", "ask_price": "10.00", "payout": "17.81"}},
        {"error": {"message": "Unknown contract proposal"}},
    ]
    with pytest.raises(RuntimeError, match="Unknown contract proposal"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 10.0)
    assert mock_ws.send.await_count == 2


@pytest.mark.asyncio
async def test_direct_buy_without_contract_response_is_reported(trade_handler, mock_ws, monkeypatch):
    from src.infrastructure.handlers import trade_handler_buy_recovery

    monkeypatch.setattr(trade_handler_buy_recovery.time, "time", lambda: 620)
    mock_ws.send.side_effect = [
        {"proposal": {"id": "quote_r25", "ask_price": "10.00", "payout": "17.81"}},
        {"error": {"message": "Unknown contract proposal"}},
        {"msg_type": "buy"},
    ]
    params = {"_indicator_direct_buy_fallback": True, "_indicator_candle_epoch": 300}
    with pytest.raises(RuntimeError, match="resposta sem contrato"):
        await trade_handler.buy_with_parameters("R_25", TradeDirection.PUT, 10.0, params=params)


@pytest.mark.asyncio
async def test_final_buy_still_blocks_negative_ev_with_same_payout(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": {"id": "quote_783", "ask_price": "10.00", "payout": "17.83"}}
    params = {
        "duration": 5,
        "duration_unit": "m",
        "min_payout_rate": 0.0,
        "_quote_guard_side_probability": 0.53247,
        "_quote_guard_min_edge": 0.0,
        "_quote_guard_safety_margin": 0.01,
    }
    with pytest.raises(RuntimeError, match=r"quote_ev=-0\.0506.*payout_rate=0\.7830"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proposal",
    [
        {"id": "new_quote"},
        {"id": "new_quote", "ask_price": "10.00"},
    ],
)
async def test_final_buy_proposal_requires_explicit_price_and_payout(trade_handler, mock_ws, proposal):
    mock_ws.send.return_value = {"proposal": proposal}
    params = {"_quote_guard_side_probability": 0.62, "_quote_guard_min_edge": 0.01}
    with pytest.raises(RuntimeError, match="sem preco ou payout"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
    mock_ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_trade_handler_fetch_proposal_payout_failures(trade_handler, mock_ws):
    mock_ws.send.return_value = {"error": {"message": "Invalid symbol"}}
    rate = await trade_handler.fetch_proposal_payout("INVALID", TradeDirection.CALL, 10.0)
    assert rate is None

    mock_ws.send.return_value = {"proposal": "invalid"}
    rate2 = await trade_handler.fetch_proposal_payout("1HZ75V", TradeDirection.CALL, 10.0)
    assert rate2 is None

    mock_ws.send.return_value = {"proposal": {"id": "no_price", "payout": "18.50"}}
    assert await trade_handler.fetch_proposal_payout("1HZ75V", TradeDirection.CALL, 10.0) is None

    mock_ws.send.return_value = {"proposal": {"id": "nan_price", "ask_price": "nan", "payout": "18.50"}}
    assert await trade_handler.fetch_proposal_payout("1HZ75V", TradeDirection.CALL, 10.0) is None

    mock_ws.send.side_effect = ConnectionError("WS dropped")
    rate3 = await trade_handler.fetch_proposal_payout("1HZ75V", TradeDirection.CALL, 10.0)
    assert rate3 is None


def test_rise_fall_contract_type_ignores_removed_barrier_settings():
    assert resolve_api_contract_type(TradeDirection.CALL, {"contract_type": "CALL"}) == "CALL"
    assert resolve_api_contract_type(TradeDirection.PUT, {"contract_type": "CALL"}) == "PUT"
    request = build_proposal_request(
        "1HZ75V", TradeDirection.CALL, 5.0, {"contract_type": "CALL", "barrier": "+1.20", "duration": 5}
    )
    assert request["contract_type"] == "CALL"
    assert "barrier" not in request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proposal",
    [
        {"id": "p", "ask_price": "bad"},
        {"id": "p", "ask_price": 0.0},
        {"id": "p", "ask_price": 10.0, "payout": -1.0},
        {"id": "p", "ask_price": 10.0, "payout": 0.0},
    ],
)
async def test_trade_handler_rejects_invalid_proposal_prices(trade_handler, mock_ws, proposal):
    mock_ws.send.return_value = {"proposal": proposal}
    with pytest.raises(RuntimeError, match="Cotacao final Rise/Fall sem preco ou payout valido"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0)


@pytest.mark.asyncio
async def test_final_buy_proposal_blocks_quote_safety_margin(trade_handler, mock_ws):
    mock_ws.send.return_value = {"proposal": {"id": "quote_sm", "ask_price": "10.00", "payout": "18.00"}}
    params = {
        "duration": 5,
        "duration_unit": "m",
        "_quote_guard_side_probability": 0.56,
        "_quote_guard_min_edge": 0.0,
        "_quote_guard_safety_margin": 0.03,
    }
    with pytest.raises(RuntimeError, match="perdeu margem de seguranca sobre break-even"):
        await trade_handler.buy_with_parameters("1HZ75V", TradeDirection.CALL, 10.0, params=params)
    mock_ws.send.assert_awaited_once()
