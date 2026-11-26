"""Integracao do buffer real M5 com cadencia, identidade e executor simulado."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.reconnect_cycle_release import release_trading_cycle_after_reconnect
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.stream_handler import StreamHandler
from tests.unit.application.test_laya_continuous_orchestrator import _m5_candles


@pytest.fixture
def pipeline(monkeypatch):
    """Monta o ciclo real, substituindo somente rede, inferencia e persistencia."""
    bars = _m5_candles()
    clock = [bars[-1].epoch + 300]
    monkeypatch.setattr("time.time", lambda: clock[0])
    stream = StreamHandler(
        MagicMock(), ["R_10"], {"granularity": 86400, "micro_granularity": 300, "mini_granularity": 300}
    )
    stream.micro_candles["R_10"] = bars.copy()
    stream.macro_candles["R_10"] = [replace(bar, close=9000) for bar in bars]
    stream.is_synchronized = True
    decision = LayaDecision("R_10", TradeDirection.CALL, 0.6, 0.6, 0.4, 0.6, 1, {})
    orch = SimpleNamespace(
        anchor="R_10",
        symbols=["R_10"],
        config={"laya": {}, "orchestrator": {"signature_boundary_seconds": 300}},
        stream=stream,
        ws=SimpleNamespace(is_running=True),
        state=SimpleNamespace(balance=1000, active_contracts={}),
        laya=SimpleNamespace(decide=AsyncMock(return_value=decision)),
        trade_handler=SimpleNamespace(fetch_proposal_snapshot=AsyncMock(return_value={"rate": 0.8})),
        risk_manager=SimpleNamespace(pending_loss=0.0),
        executor=SimpleNamespace(execute_laya_order=AsyncMock(return_value=True)),
        logger=MagicMock(),
        is_trading=False,
        running=True,
        _cycle_seq=0,
        _last_evaluated_boundary_epoch=0,
        last_data_signature="",
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.process_redis_settlement_queue", AsyncMock()
    )
    monkeypatch.setattr("src.application.services.orchestrator.trading_cycle_entry.flush_otlp_spans_async", AsyncMock())
    return orch, clock, bars


@pytest.mark.asyncio
async def test_real_stream_m5_snapshot_survives_inference_and_reconnect(pipeline):
    orch, clock, bars = pipeline
    original = orch.laya.decide.return_value

    async def advancing_model(symbol, candles, indicators, config):
        """Simula uma nova vela recebida durante a inferencia."""
        assert candles[-1] == bars[-1]
        assert indicators["close"] == bars[-1].close
        orch.stream.micro_candles["R_10"].append(replace(bars[-1], epoch=clock[0]))
        clock[0] += 300
        return original

    orch.laya.decide.side_effect = advancing_model
    assert await run_trading_cycle_if_ready(orch)
    assert orch.executor.execute_laya_order.await_args.args[2] == bars[-1].epoch
    assert orch._last_processed_epoch == bars[-1].epoch
    assert orch._last_evaluated_boundary_epoch == bars[-1].epoch + 300
    # A assinatura foi capturada apos a inferencia; outra vela exige uma nova fronteira.
    release_trading_cycle_after_reconnect(orch)
    orch.stream.micro_candles["R_10"] = bars.copy()
    assert not await run_trading_cycle_if_ready(orch)
    assert orch.laya.decide.await_count == 1
    orch.laya.decide.side_effect = None
    orch.stream.micro_candles["R_10"].append(replace(bars[-1], epoch=clock[0] - 300))
    assert await run_trading_cycle_if_ready(orch)
    assert orch.executor.execute_laya_order.await_args.args[2] == clock[0] - 300


@pytest.mark.asyncio
async def test_warmup_and_same_boundary_do_not_infer(pipeline):
    orch, clock, bars = pipeline
    orch.stream.micro_candles["R_10"] = bars[:50]
    assert not await run_trading_cycle_if_ready(orch)
    orch.laya.decide.assert_not_awaited()
    orch.stream.micro_candles["R_10"] = bars
    assert await run_trading_cycle_if_ready(orch)
    assert not await run_trading_cycle_if_ready(orch)
    clock[0] += 300
    assert not await run_trading_cycle_if_ready(orch)
    assert orch.executor.execute_laya_order.await_count == 1
