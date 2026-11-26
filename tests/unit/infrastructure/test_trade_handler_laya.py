"""Testes unitarios completos para TradeHandler e submodulos de proposta/compra."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models.trade import TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import TradeHandler
from src.infrastructure.handlers.trade_handler_buy_recovery import recover_unknown_contract_proposal
from src.infrastructure.handlers.trade_handler_errors import PurchaseOutcomeUnknownError
from src.infrastructure.handlers.trade_handler_proposal_quote import (
    contract_profit_rate,
    fetch_proposal_snapshot,
    resolve_order_proposal,
)
from src.infrastructure.handlers.trade_handler_purchase_audit import record_purchase_audit
from src.infrastructure.handlers.trade_handler_quote_guard import validate_quote_guard_params
from src.infrastructure.handlers.trade_handler_requests import (
    _contract_duration_seconds,
    build_proposal_request,
    require_current_signal_candle,
    resolve_api_contract_type,
)


def _handler(ws=None, auth=None, market_writer=None):
    ws = ws or MagicMock(request_timeout=10, send=AsyncMock())
    cfg = {
        "trading": {"mode": "demo"},
        "risk_management": {"params": {"duration": 5, "duration_unit": "m", "stake_min": 1.0}},
        "orchestrator": {"execution": {"require_quote_edge": False}},
    }
    return TradeHandler(ws, cfg, auth=auth, market_writer=market_writer)


def test_requests_and_duration_and_quote_guard_helpers():
    assert resolve_api_contract_type(TradeDirection.CALL, {"contract_type": "MULTIPLIER"}) == "MULTUP"
    assert resolve_api_contract_type(TradeDirection.PUT, {"contract_type": "MULTIPLIER"}) == "MULTDOWN"
    assert resolve_api_contract_type(TradeDirection.CALL, {}) == "CALL"

    req = build_proposal_request(
        "R_10",
        TradeDirection.CALL,
        5.0,
        {
            "contract_type": "MULTIPLIER",
            "multiplier": 50,
            "barrier": "+1",
            "cancellation": "1h",
            "limit_order": {"take_profit": 1},
        },
    )
    assert req["contract_type"] == "MULTUP" and req["multiplier"] == 50 and "limit_order" in req

    assert _contract_duration_seconds({"duration": 5, "duration_unit": "m"}) == 300
    assert _contract_duration_seconds({"duration": 10, "duration_unit": "s"}) == 10
    assert _contract_duration_seconds({"duration": 5, "duration_unit": "t"}) == 10
    assert _contract_duration_seconds({"duration": 1, "duration_unit": "d"}) == 86400
    assert _contract_duration_seconds({"duration": 2, "duration_unit": "x"}) == 120

    require_current_signal_candle({})
    with pytest.raises(RuntimeError, match="Periodo do sinal invalido"):
        require_current_signal_candle({"_signal_candle_epoch": 1, "_indicator_period_seconds": 60})
    with patch("src.infrastructure.handlers.trade_handler_requests.time.time", return_value=1764000010.0):
        expected = (1764000010 // 300) * 300 - 300
        require_current_signal_candle({"_signal_candle_epoch": expected, "_indicator_period_seconds": 300})
        with pytest.raises(RuntimeError, match="Sinal expirou"):
            require_current_signal_candle({"_signal_candle_epoch": expected - 300, "_indicator_period_seconds": 300})

    validate_quote_guard_params({"_indicator_demo_pilot": True, "_indicator_model_version": "pullback-1-fixed"}, {})
    validate_quote_guard_params({}, {"require_quote_edge": False})
    with pytest.raises(RuntimeError, match="sem probabilidade valida"):
        validate_quote_guard_params({}, {"require_quote_edge": True})
    with pytest.raises(RuntimeError, match="sem probabilidade valida"):
        validate_quote_guard_params({"_quote_guard_side_probability": "bad"}, {})
    with pytest.raises(RuntimeError, match="sem probabilidade valida"):
        validate_quote_guard_params({"_quote_guard_side_probability": 1.5}, {})
    validate_quote_guard_params({"_quote_guard_side_probability": 0.75}, {})


@pytest.mark.asyncio
async def test_proposal_quote_and_buy_recovery_and_audit():
    assert contract_profit_rate(19.5, 10.0) == pytest.approx(0.95)
    assert contract_profit_rate(5.0, 10.0) is None

    h = _handler()
    h.ws.send = AsyncMock(return_value={"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}})
    assert await h.fetch_proposal_payout("R_10", TradeDirection.CALL, 10.0) == pytest.approx(0.95)
    assert (await h.fetch_proposal_snapshot("R_10", TradeDirection.CALL, 10.0))["proposal_id"] == "p1"

    h.ws.send = AsyncMock(return_value={"proposal": {"id": "p2", "ask_price": 10.0, "payout": 9.0}})
    assert await fetch_proposal_snapshot(h, "R_10", TradeDirection.CALL, 10.0) is None
    h.ws.send = AsyncMock(return_value={"proposal": "not-a-dict"})
    assert await fetch_proposal_snapshot(h, "R_10", TradeDirection.CALL, 10.0) is None
    h.ws.send = AsyncMock(return_value={"error": {"message": "bad"}})
    assert await fetch_proposal_snapshot(h, "R_10", TradeDirection.CALL, 10.0) is None
    h.ws.send = AsyncMock(side_effect=RuntimeError("boom"))
    assert await fetch_proposal_snapshot(h, "R_10", TradeDirection.CALL, 10.0) is None

    h.ws.send = AsyncMock(return_value={"error": {"message": "bad", "code": "E1"}})
    with pytest.raises(RuntimeError, match="Erro na proposta"):
        await resolve_order_proposal(h, "R_10", TradeDirection.CALL, 10.0, {}, {"proposal": 1}, 5)

    # Buy recovery
    ws = MagicMock(send=AsyncMock(return_value={"buy": {"contract_id": 77}}))
    with patch("src.infrastructure.handlers.trade_handler_buy_recovery.time.time", return_value=1764000010.0):
        expected = (1764000010 // 300) * 300 - 300
        retry, used = await recover_unknown_contract_proposal(
            ws,
            {"error": {"message": "Unknown contract proposal"}},
            {"proposal": 1, "amount": 10.0},
            10.0,
            timeout=5,
            enabled=True,
            expected_candle_epoch=expected,
            logger=logging.getLogger("AETH"),
        )
        assert used is True and retry["buy"]["contract_id"] == 77
        with pytest.raises(RuntimeError, match="sinal expirou"):
            await recover_unknown_contract_proposal(
                ws,
                {"error": {"message": "Unknown contract proposal"}},
                {"proposal": 1},
                10.0,
                timeout=5,
                enabled=True,
                expected_candle_epoch=expected - 300,
                logger=logging.getLogger("AETH"),
            )

    # Purchase audit
    await record_purchase_audit(None, logging.getLogger("AETH"), {}, {}, "R_10", "CALL", 1, 2)
    bad_writer = SimpleNamespace(enqueue_contract_audit=AsyncMock(side_effect=RuntimeError("db down")))
    await record_purchase_audit(bad_writer, logging.getLogger("AETH"), {}, {"contract_id": 1}, "R_10", "CALL", 1, 2)


@pytest.mark.asyncio
async def test_trade_handler_ws_and_rest_buy_paths():
    writer = SimpleNamespace(enqueue_contract_audit=AsyncMock())
    h = _handler(market_writer=writer)
    h.schedule_profit_table_audit(MagicMock())

    # Compra WS feliz
    h.ws.send = AsyncMock(
        side_effect=[
            {"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}},
            {"buy": {"contract_id": 123, "buy_price": 10.0, "payout": 19.5}},
        ]
    )
    c = await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)
    assert c.contract_id == 123 and c.status == TradeStatus.OPEN

    # Compra com fallback direct_parameters=True
    with patch(
        "src.infrastructure.handlers.trade_handler.recover_unknown_contract_proposal",
        new=AsyncMock(return_value=({"buy": {"contract_id": 124, "buy_price": 10.0, "payout": 19.5}}, True)),
    ):
        h.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}},
                {"error": {"message": "Unknown contract proposal"}},
            ]
        )
        c_fb = await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)
        assert c_fb.proposal_id == "direct_parameters"

    # Timeout na compra levanta PurchaseOutcomeUnknownError
    h.ws.send = AsyncMock(
        side_effect=[
            {"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}},
            TimeoutError("timeout"),
        ]
    )
    with pytest.raises(PurchaseOutcomeUnknownError):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    # Erros de ask_price / payout / buy error / buy sem contrato
    h.ws.send = AsyncMock(return_value={"proposal": {"id": "p1", "ask_price": -1.0, "payout": 19.5}})
    with pytest.raises(RuntimeError, match="sem preco valido"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    h.ws.send = AsyncMock(return_value={"proposal": {"id": "p1", "ask_price": 10.0, "payout": -5.0}})
    with pytest.raises(RuntimeError, match="sem payout valido"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    h.ws.send = AsyncMock(
        side_effect=[
            {"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}},
            {"error": {"message": "Insufficient balance"}},
        ]
    )
    with pytest.raises(RuntimeError, match="Insufficient balance"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    h.ws.send = AsyncMock(
        side_effect=[
            {"proposal": {"id": "p1", "ask_price": 10.0, "payout": 19.5}},
            {"buy": "bad"},
        ]
    )
    with pytest.raises(RuntimeError, match="resposta sem contrato"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    # REST bulk purchase paths
    h.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="exige AuthManager"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)
    with pytest.raises(RuntimeError, match="REST sem cotacao"):
        await h.buy_with_parameters("R_10", TradeDirection.CALL, 10.0, {"_quote_guard_side_probability": 0.6})

    auth = MagicMock(mode="demo", account_id_override="VRTC1")
    auth.get_pat.return_value = ""
    h_rest = _handler(auth=auth)
    h_rest.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="AETHER_DERIV_PAT ausente"):
        await h_rest.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    auth.get_pat.return_value = "pat_ok"
    auth.account_id_override = ""
    with pytest.raises(RuntimeError, match="deriv_account_id ausente"):
        await h_rest.buy_with_parameters("R_10", TradeDirection.CALL, 10.0)

    auth.account_id_override = "VRTC1"
    rest_client = MagicMock(
        bulk_purchase=AsyncMock(return_value={"contract_id": 555, "buy_price": 10.0, "payout": 19.5})
    )
    auth.rest_client.return_value = rest_client
    c_rest = await h_rest.buy_with_parameters("R_10", TradeDirection.PUT, 10.0)
    assert c_rest.contract_id == 555
