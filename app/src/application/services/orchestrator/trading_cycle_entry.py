"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_deferred_train import try_enqueue_next_bootstrap_training
from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, resolve_engine_mode
from src.application.services.orchestrator.orchestrator_state_restore import mark_bar_processed
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


def _stop_win_blocks_cycle(orch: Any) -> bool:
    """True quando a meta diaria de lucro ja foi atingida ou o motor encerrou por stop win."""
    if getattr(orch, "shutdown_reason", None) == "stop_win":
        return True
    risk_manager = getattr(orch, "risk_manager", None)
    if risk_manager is None:
        return False
    config = getattr(orch, "config", {}) or {}
    risk_cfg = config.get("risk_management", {}) if isinstance(config, dict) else {}
    target = resolve_stop_win_target(risk_cfg, float(risk_manager.initial_bankroll))
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
    if getattr(orch, "_reconciliation_pending", False):
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
        if hasattr(orch, "get_data_state_signature") and hasattr(orch, "last_data_signature"):
            sig = orch.get_data_state_signature()
            if sig and sig == orch.last_data_signature:
                return False  # pragma: no cover
        last_epoch = getattr(orch, "_last_epoch", 0)
        last_processed = getattr(orch, "_last_processed_epoch", 0)
        if (
            isinstance(last_epoch, (int, float))
            and isinstance(last_processed, (int, float))
            and last_epoch > 0
            and last_processed == last_epoch
        ):
            return False  # pragma: no cover
        if sig and hasattr(orch, "last_data_signature"):
            orch.last_data_signature = sig

    return True


def prepare_orchestrator_run_loop(orch: Any) -> None:
    """Inicializa estado do loop principal apos streams e banner de decisao."""
    orch._last_cluster_cycle_end = time.time()
    orch.running = True
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
    """Reserva o slot de ciclo ativo; False se outro ciclo ja esta em andamento."""
    async with orch.lock:
        if _stop_win_blocks_cycle(orch):
            return False
        if orch.is_trading:
            return False
        orch.is_trading = True
    return True


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
            orch.logger.debug(
                "[C%04d] CICLO: coletando decisoes DL (%d simbolos)",
                orch._active_cycle_id,
                len(orch.symbols),
            )
            decisions = await collect_deep_learning_decisions(orch)
            if (
                int(orch._cycle_seq)
                % max(
                    1, int((orch.config.get("infra", {}).get("triton", {}) or {}).get("correlation_refresh_cycles", 5))
                )
                == 0
            ):
                await refresh_correlation_cache(orch)
            await orch.executor.execute_cluster(decisions)
    except Exception as e:
        orch.logger.error(f"FALHA: Ciclo: {e}")
        ran = True
    finally:
        orch.is_trading = False
    return ran
