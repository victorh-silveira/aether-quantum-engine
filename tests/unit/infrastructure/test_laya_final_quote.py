"""Nenhuma compra passa por cotacao invalida ou cap financeiro desatualizado."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.laya_quote_sizing import resize_laya_proposal
from src.infrastructure.handlers.trade_handler_proposal_quote import resolve_order_proposal
from tests.unit.infrastructure.test_trade_handler_laya import _handler


@pytest.mark.asyncio
async def test_final_quote_recalculates_amount_and_emits_audit(caplog):
    handler = SimpleNamespace(ws=SimpleNamespace(send=AsyncMock(return_value={"balance": {"balance": 1000}})))
    resolver = AsyncMock(side_effect=[("p1", 20, 36, {}), ("p2", 10, 18, {})])
    synced_balances: list[float] = []
    params = {"_laya_sizing": {"probability": 0.5, "pending": 0, "config": {}, "on_balance": synced_balances.append}}
    with caplog.at_level("INFO", logger="AETH"):
        result = await resize_laya_proposal(
            handler, "R_10", TradeDirection.CALL, 20, params, {"amount": 20}, 10, resolver
        )
    assert result[0] == "p2"
    assert synced_balances == [1000.0, 1000.0]
    assert resolver.await_args.args[5]["amount"] == 10
    assert "source=FLOOR" in caplog.text and "RECOVERY | pending=$0.00" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balance,proposal,match",
    [
        (0, ("p", 10, 18, {}), "saldo atual"),
        (1000, ("", 10, 18, {}), "payout ou ID"),
        (1000, ("p", 10, None, {}), "payout ou ID"),
        (1000, ("p", 20, 36, {}), "instavel"),
    ],
)
async def test_bad_final_quotes_never_buy(balance, proposal, match):
    handler = SimpleNamespace(ws=SimpleNamespace(send=AsyncMock(return_value={"balance": {"balance": balance}})))
    params = {"_laya_sizing": {"probability": 0.5, "pending": 0, "config": {}}}
    with pytest.raises(RuntimeError, match=match):
        await resize_laya_proposal(
            handler, "R_10", TradeDirection.CALL, 10, params, {}, 10, AsyncMock(return_value=proposal)
        )


@pytest.mark.asyncio
async def test_low_payout_never_buys():
    handler = SimpleNamespace(ws=SimpleNamespace(send=AsyncMock(return_value={"balance": {"balance": 1000}})))
    params = {"_laya_sizing": {"probability": 0.5, "pending": 0, "config": {"params": {"min_payout_rate": 0.75}}}}
    with pytest.raises(RuntimeError, match="payout abaixo do minimo"):
        await resize_laya_proposal(
            handler, "R_10", TradeDirection.CALL, 10, params, {}, 10, AsyncMock(return_value=("p", 10, 16.5, {}))
        )


@pytest.mark.asyncio
async def test_real_proposal_router_reads_balance_then_validates_quote():
    handler = SimpleNamespace(
        ws=SimpleNamespace(
            send=AsyncMock(
                side_effect=[{"balance": {"balance": 1000}}, {"proposal": {"id": "p", "ask_price": 10, "payout": 18}}]
            )
        )
    )
    params = {"_laya_sizing": {"probability": 0.5, "pending": 0, "config": {}}}
    assert (await resolve_order_proposal(handler, "R_10", TradeDirection.PUT, 10, params, {}, 10))[0] == "p"


@pytest.mark.asyncio
async def test_buy_uses_final_proposal_and_records_model_version():
    writer = SimpleNamespace(enqueue_contract_audit=AsyncMock())
    handler = _handler(market_writer=writer)
    handler.ws.send = AsyncMock(
        side_effect=[
            {"balance": {"balance": 1000}},
            {"proposal": {"id": "p", "ask_price": 10, "payout": 18}},
            {"buy": {"contract_id": 777, "buy_price": 10, "payout": 18}},
        ]
    )
    params = {
        "_laya_sizing": {"probability": 0.5, "pending": 0, "config": {}},
        "_indicator_model_version": "approved-test",
        "duration": 5,
    }
    result = await handler.buy_with_parameters("R_10", TradeDirection.PUT, 10, params)
    assert result.buy_price == 10
    assert handler.ws.send.call_args.args[0] == {"buy": "p", "price": 10}
    row = writer.enqueue_contract_audit.await_args.args[0]
    assert row["model_version"] == "approved-test" and row["signal_prob"] == 0.5
    handler.trading_transport = "rest"
    with pytest.raises(RuntimeError, match="WSS verificavel"):
        await handler.buy_with_parameters("R_10", TradeDirection.PUT, 10, params)


@pytest.mark.asyncio
async def test_missing_prequote_is_technical_error():
    from src.application.services.orchestrator.laya_sizing import quote_laya_sizing

    orch = SimpleNamespace(
        config={}, trade_handler=SimpleNamespace(fetch_proposal_snapshot=AsyncMock(return_value=None))
    )
    with pytest.raises(RuntimeError, match="preliminar invalida"):
        await quote_laya_sizing(orch, SimpleNamespace(symbol="R_10", direction=TradeDirection.CALL))


@pytest.mark.asyncio
async def test_order_passes_financial_context_to_final_quote(monkeypatch):
    from src.application.services.orchestrator.execution_orders import place_order
    from src.domain.risk.risk_manager import RiskManager

    handler = SimpleNamespace(
        buy_with_parameters=AsyncMock(return_value=SimpleNamespace(contract_id=1, buy_price=10, payout=18))
    )
    orch = SimpleNamespace(
        config={},
        risk_manager=RiskManager(),
        trade_handler=handler,
        ws=SimpleNamespace(),
        state=SimpleNamespace(balance=0.0),
    )
    executor = SimpleNamespace(orch=orch)
    monkeypatch.setattr(
        "src.application.services.orchestrator.execution_orders._subscribe_open_contract_background", AsyncMock()
    )
    await place_order(
        executor, "R_10", TradeDirection.CALL, 10, metrics={"laya_probability": 0.5, "laya_model_version": "test-model"}
    )
    params = handler.buy_with_parameters.await_args.kwargs["params"]
    assert params["_laya_sizing"]["probability"] == 0.5
    assert params["_indicator_model_version"] == "test-model"
    params["_laya_sizing"]["on_balance"](1234.56)
    assert orch.state.balance == 1234.56
