"""Entrada do ciclo continuo M5 no R_10 com decisao binaria 100% Laya e SKIP configuravel."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.orchestrator.cycle_indicators_logger import log_laya_cycle_decision
from src.application.services.orchestrator.laya_sizing import quote_laya_sizing
from src.application.services.orchestrator.orchestrator_state_restore import mark_bar_processed
from src.application.services.orchestrator.session_persistence_barrier import session_persistence_blocks_trading_cycle
from src.application.services.orchestrator.settlement_queue_ops import process_redis_settlement_queue
from src.application.services.orchestrator.trading_cycle_entry_guards import (
    commit_trading_cycle_data_signature,
    mark_boundary_cycle_evaluated,
    mark_cycle_attempt_complete,
    trading_cycle_entry_allowed,
)
from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.risk.kelly_recovery import sizing_from_config
from src.domain.services.laya_skip_policy import evaluate_laya_skip
from src.infrastructure.telemetry.otel_tracer import flush_otlp_spans_async, trace_span
from src.presentation.terminal.log_context import bind_log_context, clear_log_context


__all__ = ("trading_cycle_entry_allowed", "prepare_orchestrator_run_loop", "run_trading_cycle_if_ready")


def prepare_orchestrator_run_loop(orch: Any) -> None:
    """Inicializa estado do loop continuo apos conexao e sincronia de velas M5."""
    orch._last_cluster_cycle_end = 0.0
    orch._cooldown_until = 0.0
    orch._api_maintenance_until = 0.0
    orch._api_maintenance_logged_until = 0.0
    orch._session_persistence_write_active = False
    orch._stream_warmed_up_at = 0.0
    orch._warm_up_logged_until = 0.0
    orch._settlement_wait_logged = False
    orch.running = True
    skip_enabled = bool(
        (getattr(orch, "config", None) or {}).get("laya", {}).get("skip_policy", {}).get("enabled", False)
    )
    orch.logger.info(
        "INIT: Laya System 1 ativo | simbolo=%s | timeframe=M5 (300s) | skip_policy=%s",
        orch.anchor,
        "ativo" if skip_enabled else "desativado",
    )


def _closed_m5_candles(orch: Any, symbol: str) -> list[Any]:
    """Extrai serie de velas M5 fechadas para o simbolo informado."""
    return list(orch.stream.get_closed_candles(symbol, 300))


async def _execute_laya_cycle(orch: Any, candles: list[Any]) -> bool:
    """Calcula indicadores, consulta Laya, registra logs e envia ordem continua."""
    symbol = str(getattr(orch, "anchor", "R_10") or "R_10")
    candles = candles[-120:]
    snapshot = compute_indicators(candles)
    decision = await orch.laya.decide(symbol, candles, snapshot.to_dict(), orch.config)
    balance = float(getattr(orch.state, "balance", 0.0) or 0.0)
    entry_delay = max(0.0, time.time() - float(candles[-1].epoch + 300))
    verdict = evaluate_laya_skip(decision, snapshot, orch.config, entry_delay_seconds=entry_delay)
    if verdict.should_skip:
        stake = (
            sizing_from_config(
                balance,
                decision.probability,
                0.78,
                float(orch.risk_manager.pending_loss),
                orch.config.get("risk_management", {}),
            ).stake
            if balance > 0.0
            else 0.0
        )
        stake_pct = (stake / balance) if balance > 0.0 else 0.0
        log_laya_cycle_decision(orch, snapshot, decision, stake, stake_pct, balance)
        orch.logger.info(
            "SKIP | %s M5 | dir=%s | prob=%.2f%% | motivo=%s",
            decision.symbol,
            decision.direction.name,
            decision.probability * 100.0,
            verdict.reason,
        )
        if (inst := getattr(orch, "business_instrumentor", None)) is not None:
            inst.record_cycle(str(symbol), is_execution=False, reason=verdict.reason)
        return False
    sizing = await quote_laya_sizing(orch, decision)
    stake = sizing.stake
    stake_pct = (stake / balance) if balance > 0.0 else 0.0
    log_laya_cycle_decision(orch, snapshot, decision, stake, stake_pct, balance)
    min_rate = float(orch.config.get("risk_management", {}).get("params", {}).get("min_payout_rate", 0.0) or 0.0)
    if min_rate > 0.0 and sizing.payout_rate < min_rate:
        reason = f"payout_baixo({sizing.payout_rate:.2f}<{min_rate:.2f})"
        orch.logger.info(
            "SKIP | %s M5 | dir=%s | prob=%.2f%% | motivo=%s",
            decision.symbol,
            decision.direction.name,
            decision.probability * 100.0,
            reason,
        )
        if (inst := getattr(orch, "business_instrumentor", None)) is not None:
            inst.record_cycle(str(symbol), is_execution=False, reason="payout_baixo")
        return False
    if stake <= 0.0 or session_persistence_blocks_trading_cycle(orch):
        return False
    try:
        executed = await orch.executor.execute_laya_order(decision, stake, candles[-1].epoch)
    except Exception as exc:
        orch.logger.error(
            "[C%04d] EXEC_ERR | %s %s: %s", orch._active_cycle_id, decision.symbol, decision.direction.name, exc
        )
        return False
    return bool(executed)


async def run_trading_cycle_if_ready(orch: Any) -> bool:
    """Executa um ciclo continuo de decisao Laya e compra Rise/Fall M5."""
    await process_redis_settlement_queue(orch)
    if not trading_cycle_entry_allowed(orch) or orch.is_trading:
        orch._last_cycle_cluster_executed = False
        return False
    orch.is_trading = True
    ran = False
    executed = False
    try:
        if not orch.ws.is_running or not orch.stream.is_synchronized:
            orch.logger.debug("STRM: aguardando sincronia...")
            return False
        candles = _closed_m5_candles(orch, orch.anchor)
        if len(candles) < 51:
            orch.logger.warning("STRM: warmup M5 | %s | %d/51 velas fechadas", orch.anchor, len(candles))
            return False
        last_epoch = max(
            int(getattr(orch, "_last_processed_epoch", 0) or 0),
            int(getattr(orch, "_last_laya_candle_epoch", 0) or 0),
        )
        if candles[-1].epoch <= last_epoch:
            return False
        orch._cycle_seq += 1
        orch._active_cycle_id = orch._cycle_seq
        bind_log_context(cycle_id=orch._active_cycle_id, symbol=orch.anchor)
        mark_boundary_cycle_evaluated(orch)
        with trace_span("aether.trading_cycle", attributes={"cycle_id": orch._active_cycle_id, "symbol": orch.anchor}):
            executed = await _execute_laya_cycle(orch, candles)
        commit_trading_cycle_data_signature(orch)
        await mark_bar_processed(orch, orch.anchor, candles[-1].epoch)
        orch._last_processed_epoch = candles[-1].epoch
        orch._last_laya_candle_epoch = candles[-1].epoch
        orch._last_cycle_cluster_executed = executed
        ran = True
        return executed
    finally:
        clear_log_context()
        orch.is_trading = False
        if ran:
            mark_cycle_attempt_complete(orch)
            await flush_otlp_spans_async()
