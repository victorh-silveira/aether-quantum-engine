"""Guardas de pre-condicao para entrada em ciclo de trade."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.orchestrator.api_maintenance_guard import api_maintenance_blocks_trading_cycle
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, resolve_engine_mode
from src.application.services.orchestrator.post_settlement_loss_cooldown import post_loss_cooldown_blocks_trading_cycle
from src.application.services.orchestrator.session_persistence_barrier import session_persistence_blocks_trading_cycle
from src.domain.risk.stop_win_target import resolve_stop_win_target


def _orchestrator_cfg(orch: Any) -> dict:
    """Retorna o bloco orchestrator da configuracao do motor."""
    chunk = orch.config.get("orchestrator") if isinstance(orch.config, dict) else {}
    return chunk if isinstance(chunk, dict) else {}


def _cycle_cadence_seconds(orch: Any) -> int:
    """Intervalo alvo entre ciclos de decisao em segundos."""
    return int(_orchestrator_cfg(orch).get("cycle_interval_seconds") or 0)


def cycle_cadence_seconds(orch: Any) -> int:
    """Intervalo alvo entre ciclos de decisao em segundos (API publica)."""
    return _cycle_cadence_seconds(orch)


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


def _orchestrator_preconditions_block(orch: Any) -> bool:
    """True quando reconciliacao, cooldown, manutencao ou persistencia bloqueiam o ciclo."""
    return (
        getattr(orch, "_reconciliation_pending", False)
        or post_loss_cooldown_blocks_trading_cycle(orch)
        or api_maintenance_blocks_trading_cycle(orch)
        or session_persistence_blocks_trading_cycle(orch)
    )


def _engine_runtime_blocks_cycle(orch: Any) -> bool:
    """True quando modo treino, shutdown, stop-win ou slot ativo impedem nova entrada."""
    return (
        resolve_engine_mode(orch.config) == ENGINE_MODE_TRAIN
        or (not getattr(orch, "running", True) and getattr(orch, "shutdown_reason", None))
        or _stop_win_blocks_cycle(orch)
        or orch.is_trading
    )


def _active_contracts_block_cycle(orch: Any) -> bool:
    """True enquanto houver contratos abertos aguardando liquidacao."""
    if not orch.state.active_contracts:
        orch._settlement_wait_logged = False
        return False
    if not orch._settlement_wait_logged:
        orch._settlement_wait_logged = True
    return True


def _macro_cadence_blocks_cycle(orch: Any) -> bool:
    """True quando o intervalo macro entre ciclos ainda nao foi atingido."""
    return not _cycle_cadence_elapsed(orch)


def _signature_epoch_blocks_cycle(orch: Any) -> bool:
    """True quando assinatura M1 e epoch de ancora nao mudaram desde o ultimo ciclo."""
    sig = None
    signature_changed = False
    if hasattr(orch, "get_data_state_signature") and hasattr(orch, "last_data_signature"):
        sig = orch.get_data_state_signature()
        signature_changed = bool(sig and sig != orch.last_data_signature)
        if sig and not signature_changed:
            return True
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
        return True
    if sig and hasattr(orch, "last_data_signature"):
        orch.last_data_signature = sig
    return False


def _non_fast_cycle_blocks(orch: Any) -> bool:
    """True quando cadencia macro ou assinatura M1 impedem disparo fora do modo rapido DL."""
    if getattr(orch, "_dl_fast_cycle", False):
        return False
    cadence = _cycle_cadence_seconds(orch)
    if cadence > 0:
        return _macro_cadence_blocks_cycle(orch)
    return _signature_epoch_blocks_cycle(orch)


def trading_cycle_entry_allowed(orch: Any) -> bool:
    """False quando o motor nao pode iniciar um novo ciclo de decisao."""
    if _orchestrator_preconditions_block(orch):
        return False
    if _engine_runtime_blocks_cycle(orch):
        return False
    if _active_contracts_block_cycle(orch):
        return False
    return not _non_fast_cycle_blocks(orch)
