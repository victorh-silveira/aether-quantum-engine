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
from src.application.services.orchestrator.cycle_indicators_logger import log_cycle_indicators
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.orchestrator_data_signature import (
    resolve_signature_boundary_seconds,
    seconds_until_next_signature_boundary,
)
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
    """Registra indicadores M5 detalhados dos ativos ativos no ciclo."""
    log_cycle_indicators(orch)


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
    if setup_enabled:
        period_seconds = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
        target_symbol = orch.config.get("pullback_strategy", {}).get("promoted_symbol") or orch.config.get("anchor")
        status = getattr(orch, "_indicator_signal_status", {})
        signals = getattr(orch, "_indicator_period_signals", {})
        missing = (
            not any(p == period_seconds for _, p in status)
            if not target_symbol
            else (str(target_symbol), period_seconds) not in signals
        )
        if missing and time.time() % period_seconds < 20:
            await asyncio.sleep(4)
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
                        if setup_active:
                            boundary = resolve_signature_boundary_seconds(orch)
                            orch._last_m5_cycle_boundary = int(time.time() // boundary) * boundary
                            orch._last_processed_epoch = orch._last_epoch
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
