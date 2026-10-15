"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_deferred_train import try_enqueue_next_bootstrap_training
from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop
from src.application.services.execution_ema_cache import invalidate_ema_cache
from src.application.services.force_trade_mode import force_trade_from_orch
from src.application.services.indicator_consensus import refresh_indicator_consensus
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.orchestrator_data_signature import seconds_until_next_signature_boundary
from src.application.services.orchestrator.orchestrator_state_restore import mark_bar_processed
from src.application.services.orchestrator.regime_freeze_yield import await_regime_freeze_yield
from src.application.services.orchestrator.session_persistence_barrier import session_persistence_blocks_trading_cycle
from src.application.services.orchestrator.settlement_queue_ops import process_redis_settlement_queue
from src.application.services.orchestrator.trading_cycle_entry_guards import (
    _stop_win_blocks_cycle,
    commit_trading_cycle_data_signature,
    cycle_cadence_seconds,
    mark_cycle_attempt_complete,
    trading_cycle_entry_allowed,
)
from src.application.services.orchestrator.warm_up_buffer_guard import trading_cycle_warm_up_suspended
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED
from src.application.services.strategy.decision_mode import resolve_decision_mode
from src.infrastructure.market.timescale_correlation_worker import refresh_correlation_cache, start_correlation_worker
from src.presentation.terminal.log_context import bind_log_context, clear_log_context


__all__ = ("trading_cycle_entry_allowed", "prepare_orchestrator_run_loop", "run_trading_cycle_if_ready")


def prepare_orchestrator_run_loop(orch: Any) -> None:
    """Inicializa estado do loop principal apos streams e banner de decisao."""
    orch._last_cluster_cycle_end = 0.0
    orch._cooldown_until = 0.0
    orch._cooldown_skip_logged_until = 0.0
    orch._api_maintenance_until = 0.0
    orch._api_maintenance_logged_until = 0.0
    orch._session_persistence_write_active = False
    orch._stream_warmed_up_at = 0.0
    orch._warm_up_logged_until = 0.0
    orch._warm_up_waiver_applied = False
    orch._quality_guard_logged_cycle_id = -1
    orch._signature_invalidation_logged_key = ""
    orch.running = True
    orch._trading_slot_poll_task = None
    orch._dl_bootstrap_completed = (
        True if orch.config.get("pullback_strategy", {}).get("enabled", False) else prepare_inference_run_loop(orch)
    )
    mode = resolve_decision_mode(orch.config)
    emit_decision_engine_banner(orch.logger, orch.config, decision_mode=mode)
    start_correlation_worker(orch)
    if mode == "deep_learning" and not orch._dl_bootstrap_completed:
        dl_cfg = orch.config.get("deep_learning") or {}
        if bool(dl_cfg.get("online_training", False)):
            try_enqueue_next_bootstrap_training(orch)
        else:
            orch.logger.warning(
                "DL: um ou mais ativos sem checkpoint compativel; somente esses ativos ficam inelegiveis. "
                "Execute make app-train para gerar checkpoints; app-run nao treina."
            )
    orch.logger.info("")
    orch.logger.debug(
        "INIT: loop ativo | ciclo=%ds",
        cycle_cadence_seconds(orch),
    )


async def acquire_trading_cycle_lock(orch: Any) -> bool:
    """Reserva o slot de ciclo ativo sem lock bloqueante (cooperativo asyncio)."""
    if _stop_win_blocks_cycle(orch):
        return False
    if orch.is_trading:
        return False
    orch.is_trading = True
    return True


def _log_cycle_indicators(orch: Any) -> None:
    """Registra indicadores M5 detalhados do ativo ancora no ciclo."""
    if not bool(orch.config.get("pullback_strategy", {}).get("enabled", False)):
        return
    stream = getattr(orch, "stream", None) or getattr(orch, "stream_handler", None)
    if stream is None:
        return
    anchor = str(orch.config.get("anchor") or "R_25")
    candles = getattr(stream, "micro_candles", {}).get(anchor) or []
    if len(candles) < 55:
        return
    bars = candles[-128:] if len(candles) >= 128 else candles
    closes = [b.close for b in bars]
    alpha20, alpha50 = 2.0 / 21.0, 2.0 / 51.0
    fast = [closes[0]]
    for val in closes[1:]:
        fast.append(alpha20 * val + (1.0 - alpha20) * fast[-1])
    slow = [closes[0]]
    for val in closes[1:]:
        slow.append(alpha50 * val + (1.0 - alpha50) * slow[-1])
    ranges = [
        max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        for prev, curr in zip(bars[-15:-1], bars[-14:], strict=False)
    ]
    atr = sum(ranges) / 14.0 if ranges else 1.0
    pullback, confirmation = bars[-2:]
    body = abs(confirmation.close - confirmation.open)
    recuo = abs(pullback.low - fast[-2]) / atr if fast[-2] > slow[-2] else abs(pullback.high - fast[-2]) / atr
    trend = "ALTA" if fast[-2] > slow[-2] else "BAIXA"
    cid = f"C{int(getattr(orch, '_active_cycle_id', 0) or 0):04d}"
    orch.logger.info(
        "[%s] INDICADORES [%s] || CLOSE: %.2f | EMA20: %.2f | EMA50: %.2f | "
        "ATR14: %.2f | RECUO: %.2f ATR | CORPO: %.2f ATR | TENDENCIA: %s",
        cid,
        anchor,
        confirmation.close,
        fast[-2],
        slow[-2],
        atr,
        recuo,
        body / atr,
        trend,
    )


async def _execute_inference_cluster_cycle(orch: Any) -> bool:
    """Coleta inferencia DL e executa cluster quando o warm-up micro ja liberou o ciclo."""
    setup_enabled = bool(orch.config.get("pullback_strategy", {}).get("enabled", False))
    orch.logger.debug(
        "[C%04d] CICLO: consultando %s (%d simbolos no stream)",
        orch._active_cycle_id,
        "setup" if setup_enabled else "DL",
        len(orch.symbols),
    )
    cluster_executed = False
    decisions = {} if setup_enabled else await collect_deep_learning_decisions(orch)
    await refresh_indicator_consensus(orch)
    if (
        setup_enabled
        and not any(period == 300 for _, period in getattr(orch, "_indicator_signal_status", {}))
        and time.time() % 300 < 10
    ):
        await asyncio.sleep(2)
        await refresh_indicator_consensus(orch)
    _log_cycle_indicators(orch)
    if (
        int(orch._cycle_seq)
        % max(
            1,
            int(
                (orch.config.get("infra", {}).get("correlation", {}) or {}).get(
                    "correlation_refresh_cycles",
                    5,
                )
            ),
        )
        == 0
    ):
        await refresh_correlation_cache(orch)
    if not session_persistence_blocks_trading_cycle(orch):
        executed_count = await orch.executor.execute_cluster(decisions)
        cluster_executed = executed_count > 0 if isinstance(executed_count, (int, float)) else True
        await await_regime_freeze_yield(orch, decisions)
    return cluster_executed


async def run_trading_cycle_if_ready(orch: Any) -> bool:
    """Executa um ciclo completo de decisao e cluster quando permitido."""
    await process_redis_settlement_queue(orch)

    if not trading_cycle_entry_allowed(orch) or not await acquire_trading_cycle_lock(orch):
        orch._last_cycle_cluster_executed = False
        return False
    ran = False
    cluster_executed = False
    try:
        if not orch.ws.is_running or not orch.stream.is_synchronized:
            orch.logger.debug("STRM: aguardando sincronia...")
        elif resolve_decision_mode(orch.config) == "inactive":
            orch.logger.error(
                "CICLO: modo inativo; defina deep_learning.enabled=true em config/settings.json.",
            )
            ran = True
        else:
            orch._cycle_seq += 1
            if orch._cycle_seq > 1:
                orch.logger.info("")
            orch._active_cycle_id = orch._cycle_seq
            invalidate_ema_cache(orch._active_cycle_id)
            orch._side_eq_log_keys = set()
            bind_log_context(cycle_id=orch._active_cycle_id, symbol=getattr(orch, "anchor", None))
            ran = True
            if trading_cycle_warm_up_suspended(orch) != SIGNAL_SUSPENDED:
                cluster_executed = await _execute_inference_cluster_cycle(orch)
                if cluster_executed:
                    commit_trading_cycle_data_signature(orch)
                    orch._last_processed_epoch = orch._last_epoch
                    await mark_bar_processed(orch, orch.anchor, orch._last_epoch)
                else:
                    commit_trading_cycle_data_signature(orch)
                    if not force_trade_from_orch(orch):
                        boundary_wait = float(seconds_until_next_signature_boundary(orch))
                        orch_cfg = orch.config.get("orchestrator") if isinstance(orch.config, dict) else {}
                        if not isinstance(orch_cfg, dict):
                            orch_cfg = {}
                        try:
                            empty_cap = float(orch_cfg.get("exec_empty_retry_seconds", 15))
                        except (TypeError, ValueError):
                            empty_cap = 15.0
                        setup_active = bool(orch.config.get("pullback_strategy", {}).get("enabled", False))
                        if setup_active and time.time() % 300 >= 30:
                            delay = max(5.0, boundary_wait)
                        else:
                            delay = min(max(boundary_wait, 1.0), max(5.0, empty_cap))
                        orch._cooldown_until = time.time() + delay
    except Exception as e:
        orch.logger.error("FALHA: Ciclo: %s", e)
        ran = True
    finally:
        clear_log_context()
        orch.is_trading = False
        if ran:
            mark_cycle_attempt_complete(orch)
        orch._last_cycle_cluster_executed = cluster_executed
    return ran
