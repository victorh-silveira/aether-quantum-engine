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
    if target <= 0.0:
        return False
    return float(risk_manager.total_session_profit) >= target


def trading_cycle_entry_allowed(orch: Any) -> bool:
    """False quando o motor nao pode iniciar um novo ciclo de decisao."""
    if resolve_engine_mode(orch.config) == ENGINE_MODE_TRAIN:
        return False
    if not getattr(orch, "running", True) and getattr(orch, "shutdown_reason", None):
        return False
    if _stop_win_blocks_cycle(orch):
        return False
    if orch.is_trading:
        return False
    if orch.state.active_contracts:
        if not orch._settlement_wait_logged:
            orch._settlement_wait_logged = True
        return False
    orch._settlement_wait_logged = False
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
    dl = orch.config.get("deep_learning", {}) if isinstance(orch.config.get("deep_learning"), dict) else {}
    orch.logger.info(
        "INIT: Motor ativo | ciclo=%ds | threshold=%.2f/%.2f",
        int(orch_cfg.get("cycle_interval_seconds") or 0),
        float(dl.get("confidence_call_threshold", 0.75)),
        float(dl.get("confidence_put_threshold", 0.25)),
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
