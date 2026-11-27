"""Testes de integracao unitaria do ciclo continuo Laya no R_10 M5."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.cycle_indicators_logger import log_laya_cycle_decision
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_EXECUTE, resolve_engine_mode
from src.application.services.orchestrator.trading_cycle_entry import (
    prepare_orchestrator_run_loop,
    run_trading_cycle_if_ready,
)
from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection


def _m5_candles(count: int = 70) -> list[Candle]:
    base = 1_764_000_000
    return [
        Candle(
            "R_10",
            6500.0 + i,
            6501.5 + i,
            6499.0 + i,
            6501.0 + i,
            datetime.fromtimestamp(base + i * 300, tz=UTC),
            base + i * 300,
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_continuous_laya_cycle_never_skips_and_logs(monkeypatch):
    candles = _m5_candles(70)
    logger = MagicMock()
    decision = LayaDecision(
        symbol="R_10",
        direction=TradeDirection.CALL,
        probability=0.75,
        prob_call=0.75,
        prob_put=0.25,
        confidence=0.75,
        latency_ms=33.2,
        indicators={"ema_20": 6560.0},
    )
    orch = SimpleNamespace(
        anchor="R_10",
        config={
            "anchor": "R_10",
            "laya": {"min_stake_pct": 0.01, "max_stake_pct": 0.05, "stake_min": 1.0},
            "orchestrator": {"signature_boundary_seconds": 300},
        },
        state=SimpleNamespace(balance=1000.0, active_contracts={}),
        ws=SimpleNamespace(is_running=True),
        stream=SimpleNamespace(is_synchronized=True, get_closed_candles=lambda sym, g: candles),
        laya=SimpleNamespace(decide=AsyncMock(return_value=decision)),
        trade_handler=SimpleNamespace(fetch_proposal_snapshot=AsyncMock(return_value={"rate": 0.8})),
        risk_manager=SimpleNamespace(pending_loss=0.0),
        executor=SimpleNamespace(execute_laya_order=AsyncMock(return_value=True)),
        logger=logger,
        running=True,
        is_trading=False,
        _reconciliation_pending=False,
        _cycle_seq=0,
        _active_cycle_id=0,
        _last_processed_epoch=0,
        _last_evaluated_boundary_epoch=0,
        last_data_signature="",
        _last_cluster_cycle_end=0.0,
        _last_cycle_cluster_executed=False,
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.process_redis_settlement_queue",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry_guards.get_data_state_signature",
        lambda _o: "sig_1",
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.commit_trading_cycle_data_signature",
        lambda o: setattr(o, "last_data_signature", "sig_1"),
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.mark_bar_processed",
        AsyncMock(side_effect=lambda o, _sym, ep: setattr(o, "_last_processed_epoch", ep)),
    )

    prepare_orchestrator_run_loop(orch)
    assert resolve_engine_mode(orch.config) == ENGINE_MODE_EXECUTE
    ok = await run_trading_cycle_if_ready(orch)
    assert ok is True
    orch.laya.decide.assert_awaited_once()
    orch.executor.execute_laya_order.assert_awaited_once()
    stake_arg = orch.executor.execute_laya_order.await_args.args[1]
    assert stake_arg == 50.0  # Kelly positivo limitado a 5% da banca
    logged = [
        call.args[0] % call.args[1:] if len(call.args) > 1 else call.args[0] for call in logger.info.call_args_list
    ]
    assert any("INDICATORS |" in line for line in logged)
    for section in ("TENDENCIA", "MOMENTUM", "DIRECAO", "OSCILADORES", "VOLATILIDADE", "ESTRUTURA", "MODELO"):
        assert any(section in line for line in logged)
    decision_log = next(line for line in logged if "LAYA_DECISION |" in line)
    assert "P(CALL)=75.00%" in decision_log
    assert "prob=0.75 | conf=0.75" in decision_log

    snap = compute_indicators(candles)
    log_laya_cycle_decision(orch, snap, decision, 30.0, 0.03, 1000.0)


@pytest.mark.asyncio
async def test_laya_cycle_skips_when_policy_blocks(monkeypatch):
    candles = _m5_candles(70)
    logger = MagicMock()
    decision = LayaDecision(
        symbol="R_10",
        direction=TradeDirection.CALL,
        probability=0.501,
        prob_call=0.501,
        prob_put=0.499,
        confidence=0.501,
        latency_ms=20.0,
        indicators={"ema_20": 6560.0},
    )
    instrumentor = MagicMock()
    orch = SimpleNamespace(
        anchor="R_10",
        config={
            "anchor": "R_10",
            "laya": {"skip_policy": {"enabled": True, "min_probability": 0.5025}},
            "orchestrator": {"signature_boundary_seconds": 300},
        },
        state=SimpleNamespace(balance=1000.0, active_contracts={}),
        ws=SimpleNamespace(is_running=True),
        stream=SimpleNamespace(is_synchronized=True, get_closed_candles=lambda sym, g: candles),
        laya=SimpleNamespace(decide=AsyncMock(return_value=decision)),
        trade_handler=SimpleNamespace(fetch_proposal_snapshot=AsyncMock(return_value={"rate": 0.8})),
        risk_manager=SimpleNamespace(pending_loss=0.0),
        executor=SimpleNamespace(execute_laya_order=AsyncMock(return_value=True)),
        business_instrumentor=instrumentor,
        logger=logger,
        running=True,
        is_trading=False,
        _reconciliation_pending=False,
        _cycle_seq=0,
        _active_cycle_id=0,
        _last_processed_epoch=0,
        _last_evaluated_boundary_epoch=0,
        last_data_signature="",
        _last_cluster_cycle_end=0.0,
        _last_cycle_cluster_executed=False,
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.process_redis_settlement_queue",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry_guards.get_data_state_signature",
        lambda _o: "sig_1",
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.commit_trading_cycle_data_signature",
        lambda o: setattr(o, "last_data_signature", "sig_1"),
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.trading_cycle_entry.mark_bar_processed",
        AsyncMock(side_effect=lambda o, _sym, ep: setattr(o, "_last_processed_epoch", ep)),
    )

    ok = await run_trading_cycle_if_ready(orch)
    assert ok is False
    orch.trade_handler.fetch_proposal_snapshot.assert_not_awaited()
    orch.executor.execute_laya_order.assert_not_awaited()
    logged = [
        call.args[0] % call.args[1:] if len(call.args) > 1 else call.args[0] for call in logger.info.call_args_list
    ]
    assert any("SKIP | R_10 M5" in line for line in logged)
    instrumentor.record_cycle.assert_called_once()
