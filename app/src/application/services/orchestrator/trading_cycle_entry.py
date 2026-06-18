"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.deep_learning.dl_deferred_train import try_enqueue_next_bootstrap_training
from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, resolve_engine_mode
from src.application.services.strategy.decision_mode import resolve_decision_mode
from src.domain.risk.stop_win_target import resolve_stop_win_target


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

    if hasattr(orch, "get_data_state_signature") and hasattr(orch, "last_data_signature"):
        sig = orch.get_data_state_signature()
        if sig and sig == orch.last_data_signature:
            return False  # pragma: no cover

    if not getattr(orch, "_dl_fast_cycle", False):
        last_epoch = getattr(orch, "_last_epoch", 0)
        last_processed = getattr(orch, "_last_processed_epoch", 0)
        if (
            isinstance(last_epoch, (int, float))
            and isinstance(last_processed, (int, float))
            and last_epoch > 0
            and last_processed == last_epoch
        ):
            return False  # pragma: no cover

    return True


def prepare_orchestrator_run_loop(orch: Any) -> None:
    """Inicializa estado do loop principal apos streams e banner de decisao."""
    orch._last_cluster_cycle_end = time.time()
    orch.running = True
    orch._dl_bootstrap_completed = prepare_inference_run_loop(orch)
    mode = resolve_decision_mode(orch.config)
    emit_decision_engine_banner(orch.logger, orch.config, decision_mode=mode)
    if mode == "deep_learning" and not orch._dl_bootstrap_completed:
        try_enqueue_next_bootstrap_training(orch)
    orch_cfg = orch.config.get("orchestrator") if isinstance(orch.config.get("orchestrator"), dict) else {}
    orch.logger.debug(
        "INIT: loop ativo | ciclo=%ds",
        int(orch_cfg.get("cycle_interval_seconds") or 0),
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
