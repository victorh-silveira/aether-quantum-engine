"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_deferred_train import try_enqueue_next_bootstrap_training
from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop
from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from src.application.services.execution_quality_gate_starvation import (
    prepare_quality_skipped_cycles_counter,
    record_quality_guard_cycle_skip,
    reset_quality_skipped_cycles_counter_for_orch,
)
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.api_maintenance_guard import api_maintenance_blocks_trading_cycle
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, resolve_engine_mode
from src.application.services.orchestrator.execution_quality_skip_yield import (
    await_quality_skip_yield,
    sanitize_quality_skip_decisions,
)
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.orchestrator_state_restore import mark_bar_processed
from src.application.services.orchestrator.post_settlement_loss_cooldown import post_loss_cooldown_blocks_trading_cycle
from src.application.services.orchestrator.regime_freeze_yield import await_regime_freeze_yield, cluster_collect_aborted
from src.application.services.orchestrator.session_persistence_barrier import session_persistence_blocks_trading_cycle
from src.application.services.orchestrator.warm_up_buffer_guard import trading_cycle_warm_up_suspended
from src.application.services.strategy.decision_mode import resolve_decision_mode
from src.domain.risk.stop_win_target import resolve_stop_win_target
from src.infrastructure.market.timescale_correlation_worker import refresh_correlation_cache, start_correlation_worker


def _orchestrator_cfg(orch: Any) -> dict:
    """Retorna o bloco orchestrator da configuracao do motor."""
    chunk = orch.config.get("orchestrator") if isinstance(orch.config, dict) else {}
    return chunk if isinstance(chunk, dict) else {}


def _cycle_cadence_seconds(orch: Any) -> int:
    """Intervalo alvo entre ciclos de decisao em segundos."""
    return int(_orchestrator_cfg(orch).get("cycle_interval_seconds") or 0)


def _cycle_cadence_elapsed(orch: Any) -> bool:
    """True quando o tempo desde o ultimo ciclo atingiu o intervalo configurado."""
    cadence = _cycle_cadence_seconds(orch)
    if cadence <= 0:
        return False
    last_end = float(getattr(orch, "_last_cluster_cycle_end", 0.0))
    return last_end > 0.0 and (time.time() - last_end) >= cadence


def _log_market_signature_invalidation(orch: Any, *, previous: str, current: str) -> None:
    """Registra invalidacao deduplicada do cache tecnico por divergencia M1."""
    if not current or current == previous:
        return
    logged_key = str(getattr(orch, "_signature_invalidation_logged_key", "") or "")
    if logged_key == current:
        return
    orch._signature_invalidation_logged_key = current
    orch.logger.debug(
        "DATA_SIG: cache invalidado por divergencia M1 | anterior=%s | atual=%s | inferencia reinicializada",
        previous or "-",
        current,
    )


def _stop_win_blocks_cycle(orch: Any) -> bool:
    """True quando a meta diaria de lucro ou perda ja foi atingida."""
    if getattr(orch, "shutdown_reason", None) == "stop_win":
        return True
    risk_manager = getattr(orch, "risk_manager", None)
    if risk_manager is None:
        return False
    config = getattr(orch, "config", {}) or {}
    risk_cfg = config.get("risk_management", {}) if isinstance(config, dict) else {}
    persisted_target = None
    if hasattr(orch, "state_mgr") and orch.state_mgr is not None:
        persisted_target = float(orch.state_mgr.state.daily_stop_win_target)
    target = resolve_stop_win_target(
        risk_cfg,
        float(risk_manager.initial_bankroll),
        persisted_target=persisted_target if persisted_target > 0.0 else None,
    )
    pnl = float(risk_manager.total_session_profit)

    if hasattr(orch, "state_mgr") and orch.state_mgr is not None and type(orch.state_mgr).__name__ == "StateManager":
        if orch.state_mgr.state.initial_balance <= 0.0:
            orch.state_mgr.state.initial_balance = float(risk_manager.initial_bankroll)
        if orch.state_mgr.state.daily_stop_win_target <= 0.0:
            orch.state_mgr.state.daily_stop_win_target = float(target)
        if orch.state_mgr.state.total_trades_today <= 0 and pnl > 0.0:
            orch.state_mgr.state.total_trades_today = 1

        orch.state_mgr.state.current_balance = orch.state_mgr.state.initial_balance + pnl
        orch.state_mgr.check_session_limits()
        return orch.state_mgr.state.stop_win_triggered

    if target <= 0.0:  # pragma: no cover
        return False  # pragma: no cover
    return pnl >= target  # pragma: no cover


def trading_cycle_entry_allowed(orch: Any) -> bool:
    """False quando o motor nao pode iniciar um novo ciclo de decisao."""
    if (
        getattr(orch, "_reconciliation_pending", False)
        or post_loss_cooldown_blocks_trading_cycle(orch)
        or api_maintenance_blocks_trading_cycle(orch)
        or session_persistence_blocks_trading_cycle(orch)
    ):
        return False
    if (
        resolve_engine_mode(orch.config) == ENGINE_MODE_TRAIN
        or (not getattr(orch, "running", True) and getattr(orch, "shutdown_reason", None))
        or _stop_win_blocks_cycle(orch)
        or orch.is_trading
    ):
        return False

    if orch.state.active_contracts:
        if not orch._settlement_wait_logged:
            orch._settlement_wait_logged = True
        return False
    orch._settlement_wait_logged = False

    if not getattr(orch, "_dl_fast_cycle", False) and not _cycle_cadence_elapsed(orch):
        sig = None
        signature_changed = False
        if hasattr(orch, "get_data_state_signature") and hasattr(orch, "last_data_signature"):
            sig = orch.get_data_state_signature()
            signature_changed = bool(sig and sig != orch.last_data_signature)
            if sig and not signature_changed:
                return False
            if signature_changed:
                _log_market_signature_invalidation(orch, previous=orch.last_data_signature, current=sig)
        last_epoch = getattr(orch, "_last_epoch", 0)
        last_processed = getattr(orch, "_last_processed_epoch", 0)
        if (
            not signature_changed
            and isinstance(last_epoch, (int, float))
            and isinstance(last_processed, (int, float))
            and last_epoch > 0
            and last_processed == last_epoch
        ):
            return False
        if sig and hasattr(orch, "last_data_signature"):
            orch.last_data_signature = sig

    return True


def prepare_orchestrator_run_loop(orch: Any) -> None:
    """Inicializa estado do loop principal apos streams e banner de decisao."""
    orch._last_cluster_cycle_end = time.time()
    orch._cooldown_until = 0.0
    orch._cooldown_skip_logged_until = 0.0
    orch._api_maintenance_until = 0.0
    orch._api_maintenance_logged_until = 0.0
    orch._session_persistence_write_active = False
    orch._stream_warmed_up_at = 0.0
    orch._warm_up_logged_until = 0.0
    orch._quality_guard_logged_cycle_id = -1
    orch._signature_invalidation_logged_key = ""
    orch.running = True
    orch._trading_slot_poll_task = None
    orch._dl_bootstrap_completed = prepare_inference_run_loop(orch)
    mode = resolve_decision_mode(orch.config)
    emit_decision_engine_banner(orch.logger, orch.config, decision_mode=mode)
    start_correlation_worker(orch)
    if mode == "deep_learning" and not orch._dl_bootstrap_completed:
        try_enqueue_next_bootstrap_training(orch)
    orch.logger.info("")
    orch.logger.debug(
        "INIT: loop ativo | ciclo=%ds",
        _cycle_cadence_seconds(orch),
    )


async def acquire_trading_cycle_lock(orch: Any) -> bool:
    """Reserva o slot de ciclo ativo sem lock bloqueante (cooperativo asyncio)."""
    if _stop_win_blocks_cycle(orch):
        return False
    if orch.is_trading:
        return False
    orch.is_trading = True
    return True


async def _execute_inference_cluster_cycle(orch: Any) -> None:
    """Coleta inferencia DL e executa cluster quando o warm-up micro ja liberou o ciclo."""
    await prepare_quality_skipped_cycles_counter(orch)
    orch.loss_tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    orch.logger.debug(
        "[C%04d] CICLO: coletando decisoes DL (%d simbolos)",
        orch._active_cycle_id,
        len(orch.symbols),
    )
    post_lock_decisions = None
    quality_skip_pending = False
    async with orchestrator_atomic_state_context(orch):
        decisions = await collect_deep_learning_decisions(orch)
        if (
            int(orch._cycle_seq)
            % max(
                1,
                int(
                    (orch.config.get("infra", {}).get("triton", {}) or {}).get(
                        "correlation_refresh_cycles",
                        5,
                    )
                ),
            )
            == 0
        ):
            await refresh_correlation_cache(orch)
        if cluster_collect_aborted(decisions):
            post_lock_decisions = decisions
        elif quality_conviction_suspends_cluster(orch, decisions):
            sanitize_quality_skip_decisions(decisions)
            post_lock_decisions = decisions
            record_quality_guard_cycle_skip(orch)
            quality_skip_pending = any(
                isinstance(entry, dict)
                and isinstance(entry.get("metrics"), dict)
                and entry["metrics"].get("execution_gate_state") == "meta_zscore_reject"
                for entry in decisions.values()
            )
        elif not session_persistence_blocks_trading_cycle(orch):
            await orch.executor.execute_cluster(decisions)
            post_lock_decisions = decisions
            await reset_quality_skipped_cycles_counter_for_orch(orch)
    if post_lock_decisions is not None:
        if quality_skip_pending:
            await await_quality_skip_yield(orch)
        else:
            await await_regime_freeze_yield(orch, post_lock_decisions)


async def run_trading_cycle_if_ready(orch: Any) -> bool:
    """Executa um ciclo completo de decisao e cluster quando permitido."""
    if not trading_cycle_entry_allowed(orch) or not await acquire_trading_cycle_lock(orch):
        return False
    ran = False
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
            orch._last_processed_epoch = orch._last_epoch
            await mark_bar_processed(orch, orch.anchor, orch._last_epoch)
            ran = True
            if trading_cycle_warm_up_suspended(orch) != SIGNAL_SUSPENDED:
                await _execute_inference_cluster_cycle(orch)
    except Exception as e:
        orch.logger.error(f"FALHA: Ciclo: {e}")
        ran = True
    finally:
        orch.is_trading = False
    return ran
