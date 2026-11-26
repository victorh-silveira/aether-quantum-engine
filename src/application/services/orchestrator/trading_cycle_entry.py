"""Entrada do ciclo continuo M5 no R_10 com decisao binaria 100% Laya (zero SKIP)."""

from __future__ import annotations

from typing import Any

from src.application.services.orchestrator.cycle_indicators_logger import log_laya_cycle_decision
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
from src.domain.risk.laya_stake import calculate_laya_stake
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
    orch.logger.info("INIT: Laya System 1 continuo ativo | simbolo=%s | timeframe=M5 (300s) | zero SKIP", orch.anchor)


def _closed_m5_candles(orch: Any, symbol: str) -> list[Any]:
    """Extrai serie de velas M5 fechadas para o simbolo informado."""
    return list(orch.stream.get_closed_candles(symbol, 300))


async def _execute_laya_cycle(orch: Any, candles: list[Any]) -> bool:
    """Calcula indicadores, consulta Laya, registra logs e envia ordem continua."""
    symbol = str(getattr(orch, "anchor", "R_10") or "R_10")
    snapshot = compute_indicators(candles)
    decision = await orch.laya.decide(symbol, candles, snapshot.to_dict(), orch.config)
    laya_cfg = orch.config.get("laya") if isinstance(orch.config.get("laya"), dict) else {}
    balance = float(getattr(orch.state, "balance", 0.0) or 0.0)
    stake, stake_pct = calculate_laya_stake(
        balance,
        decision.probability,
        stake_min=float(laya_cfg.get("stake_min", 1.0)),
        min_stake_pct=float(laya_cfg.get("min_stake_pct", 0.01)),
        max_stake_pct=float(laya_cfg.get("max_stake_pct", 0.05)),
    )
    log_laya_cycle_decision(orch, snapshot, decision, stake, stake_pct, balance)
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
