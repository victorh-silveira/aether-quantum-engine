"""Fluxo Touch sem compras reais: proposta exata, limites e isolamento do legado."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.touch_runtime import run_touch_cycle
from src.domain.models.touch_policy import TouchPolicy
from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.tick_buffer import TickBuffer
from src.infrastructure.handlers.touch_broker import TouchQuoteRejectedError, buy_touch_quote, quote_touch_candidates
from tests.unit.application.test_touch_pipeline import _bundle, _quote


@pytest.mark.asyncio
async def test_touch_startup_does_not_load_directional_models():
    from src.application.services.orchestrator import ws_bootstrap as boot

    orch = SimpleNamespace(config={"touch": {"enabled": True}}, infra=object())
    with (
        patch.object(boot, "validate_infra_services", AsyncMock()),
        patch.object(boot, "bootstrap_meta_classifier_client", AsyncMock()) as meta,
        patch.object(boot, "bootstrap_and_validate_models", AsyncMock()) as tcn,
        patch.object(boot, "restore_orchestrator_state", AsyncMock(side_effect=RuntimeError("stop before auth"))),
        patch.object(boot, "_setup_trading_session_failure", return_value=False),
    ):
        assert not await boot.setup_trading_session(orch)
    meta.assert_not_awaited()
    tcn.assert_not_awaited()


def _orch():
    handler = SimpleNamespace(
        ws=SimpleNamespace(send=AsyncMock(), request_timeout=1),
        trading_transport="ws",
        logger=MagicMock(),
        _record_purchase_audit=AsyncMock(),
    )
    return SimpleNamespace(
        config={
            "touch": {"enabled": True},
            "risk_management": {"kelly": {"max_stake": 15.0, "kelly_fraction": 0.25}, "params": {"stake_min": 1.0}},
        },
        state=SimpleNamespace(balance=10000.0, active_contracts={}, add_contract=AsyncMock()),
        risk_manager=SimpleNamespace(
            active_contract_ids=[], contract_to_symbol={}, record_contract_stake=MagicMock(), begin_cluster=MagicMock()
        ),
        stream=SimpleNamespace(tick_buffer=SimpleNamespace(recent_ticks=MagicMock(return_value=[]))),
        trade_handler=handler,
        logger=MagicMock(),
        _active_cycle_id=1,
        _contract_cycle={},
        executor=SimpleNamespace(_run_settlement_watch=AsyncMock()),
        ws=handler.ws,
    )


@pytest.mark.asyncio
async def test_runtime_collects_without_model_and_never_falls_back():
    orch = _orch()
    with (
        patch("src.application.services.touch_runtime.quote_touch_candidates", AsyncMock(return_value=[])),
        patch("src.application.services.touch_runtime.append_touch_quotes") as journal,
        patch("src.application.services.touch_runtime.load_touch_model", side_effect=FileNotFoundError),
    ):
        assert not await run_touch_cycle(orch)
    journal.assert_called_once_with([])
    orch.trade_handler.ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_guards_and_purchase():
    orch = _orch()
    orch.state.active_contracts = {1: object()}
    assert not await run_touch_cycle(orch)
    orch.state.active_contracts.clear()
    orch._touch_buy_uncertain = True
    assert not await run_touch_cycle(orch)
    orch._touch_buy_uncertain = False
    orch.state.balance = 0.1
    assert not await run_touch_cycle(orch)
    orch.state.balance = 10000.0
    quote = _quote(features=[5.0] * 5)
    contract = SimpleNamespace(contract_id=42, buy_price=1.0)

    def spawn(_orch, coroutine, **_kwargs):
        coroutine.close()

    with (
        patch("src.application.services.touch_runtime.quote_touch_candidates", AsyncMock(return_value=[quote])),
        patch("src.application.services.touch_runtime.append_touch_quotes"),
        patch("src.application.services.touch_runtime.load_touch_model", return_value=_bundle()),
        patch("src.application.services.touch_runtime.validate_touch_bundle"),
        patch("src.application.services.touch_runtime.buy_touch_quote", AsyncMock(return_value=contract)) as buy,
        patch("src.application.services.touch_runtime.subscribe_open_contract", AsyncMock()),
        patch("src.application.services.touch_runtime.spawn_background", side_effect=spawn),
    ):
        orch.trade_handler.trading_transport = "rest"
        assert not await run_touch_cycle(orch)
        orch.trade_handler.trading_transport = "ws"
        orch.config["risk_management"]["kelly"]["kelly_fraction"] = 0.000001
        assert not await run_touch_cycle(orch)
        orch.config["risk_management"]["kelly"]["kelly_fraction"] = 0.25
        with patch("src.application.services.touch_runtime.select_touch_quote", return_value=None):
            assert not await run_touch_cycle(orch)
        buy.side_effect = TouchQuoteRejectedError("vencida")
        assert not await run_touch_cycle(orch)
        assert orch._touch_buy_uncertain is False
        buy.side_effect = TimeoutError("buy ACK desconhecido")
        with pytest.raises(TimeoutError):
            await run_touch_cycle(orch)
        assert orch._touch_buy_uncertain is True
        orch._touch_buy_uncertain = False
        buy.side_effect = ValueError("payload invalido apos compra")
        with pytest.raises(ValueError):
            await run_touch_cycle(orch)
        assert orch._touch_buy_uncertain is True
        orch._touch_buy_uncertain = False
        buy.side_effect = None
        assert await run_touch_cycle(orch)
    assert orch.risk_manager.active_contract_ids == [42]
    assert orch._contract_cycle == {42: 1}
    assert orch._touch_buy_uncertain is False


@pytest.mark.asyncio
async def test_broker_quotes_real_payout_and_absolute_barrier():
    handler = _orch().trade_handler
    policy = TouchPolicy()
    ticks = np.column_stack((np.arange(1000, 302000, 1000), 100 + np.sin(np.arange(301) / 10)))
    handler.ws.send.return_value = {"proposal": {"id": "q", "ask_price": "1", "payout": "2", "spot_time": 301}}
    with patch("src.infrastructure.handlers.touch_broker.time.time", return_value=301.0):
        quotes = await quote_touch_candidates(handler, ticks, 1.0, policy)
        assert len(quotes) == 8
        assert {q["contract_type"] for q in quotes} == {"ONETOUCH", "NOTOUCH"}
        for call in handler.ws.send.await_args_list:
            req = call.args[0]
            assert req["duration"] == 300 and float(req["barrier"]) > 0
            assert not req["barrier"].startswith("+")
        handler.ws.send.return_value = {"error": {"message": "barrier unsupported"}}
        assert await quote_touch_candidates(handler, ticks, 1.0, policy) == []
        handler.ws.send.return_value = {"proposal": {"id": "q", "ask_price": 2.0, "payout": 3.0, "spot_time": 301}}
        with pytest.raises(ValueError, match="teto"):
            await quote_touch_candidates(handler, ticks, 1.0, policy)
        handler.ws.send.return_value["proposal"]["spot_time"] = 1
        with pytest.raises(ValueError, match="Timestamp"):
            await quote_touch_candidates(handler, ticks, 1.0, policy)
    with (
        patch("src.infrastructure.handlers.touch_broker.time.time", return_value=400.0),
        pytest.raises(ValueError, match="desatualizados"),
    ):
        await quote_touch_candidates(handler, ticks, 1.0, policy)


@pytest.mark.asyncio
async def test_buy_exact_proposal_without_retry_or_rest_fallback():
    handler = _orch().trade_handler
    quote = _quote(decision_ms=1000000)
    handler.ws.send.return_value = {"buy": {"contract_id": 42, "start_time": 1000, "buy_price": 1.0}}
    with patch("src.infrastructure.handlers.touch_broker.time.time", return_value=1001.0):
        contract = await buy_touch_quote(handler, quote, TouchPolicy())
        assert contract.direction == TradeDirection.ONETOUCH
        assert contract.expiry_time == 1300
        handler.ws.send.assert_awaited_once_with({"buy": "q", "price": 1.0}, timeout=1)
        handler.trading_transport = "rest"
        with pytest.raises(ValueError, match="WSS"):
            await buy_touch_quote(handler, quote, TouchPolicy())
        handler.trading_transport = "ws"
        with pytest.raises(ValueError, match="vencida"):
            await buy_touch_quote(handler, {**quote, "decision_ms": 1}, TouchPolicy())
        handler.ws.send.return_value = {"error": {"message": "closed"}}
        with pytest.raises(RuntimeError, match="rejeitada"):
            await buy_touch_quote(handler, quote, TouchPolicy())


def test_tick_history_survives_bar_close_but_not_disconnect():
    buffer = TickBuffer(["1HZ75V"])
    buffer.record_tick("1HZ75V", 1000, 100.0)
    buffer.on_bar_close("1HZ75V", 0)
    assert buffer.recent_ticks("1HZ75V") == [(1000, 100.0)]
    assert buffer.recent_ticks("UNKNOWN") == []
    buffer.reset_live_accumulators()
    assert buffer.recent_ticks("1HZ75V") == []


@pytest.mark.asyncio
async def test_cycle_routes_touch_without_directional_collector():
    from src.application.services.orchestrator.trading_cycle_entry import _execute_inference_cluster_cycle

    with patch(
        "src.application.services.orchestrator.trading_cycle_entry.run_touch_cycle", AsyncMock(return_value=True)
    ):
        assert await _execute_inference_cluster_cycle(_orch())


def test_touch_banner_is_not_directional():
    from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner

    logger = MagicMock()
    emit_decision_engine_banner(logger, {"touch": {"enabled": True}}, decision_mode="deep_learning")
    assert "TOUCH/NOTOUCH" in logger.info.call_args.args[0]


def test_touch_settlement_does_not_train_directional_models():
    from src.application.services.orchestrator.settlement_outcome import process_contract_outcome

    orch = MagicMock()
    orch.state.balance = 100.0
    orch.risk_manager.contract_to_symbol = {42: "1HZ75V"}
    orch.risk_manager.contract_stakes = {42: 1.0}
    orch.risk_manager.contract_requested_stakes = {}
    orch.risk_manager.active_contract_ids = [42]
    orch._contract_cycle = {42: 1}
    orch._cluster_results = []
    orch._session_wins = 0
    contract = SimpleNamespace(direction=TradeDirection.NOTOUCH, stake=1.0, buy_price=1.0)
    with (
        patch("src.application.services.orchestrator.settlement_outcome._record_directional_learning") as learn,
        patch("src.application.services.orchestrator.settlement_outcome.record_symbol_outcome"),
    ):
        process_contract_outcome(orch, {"buy_price": 1.0}, contract, 42, 1.0, log_cluster_summary=MagicMock())
    learn.assert_not_called()
    orch.risk_manager.register_result.assert_called_once()
    assert orch._session_wins == 1
