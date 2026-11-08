"""Barreira atomica de persistencia pos-reset linear D'Alembert."""

from __future__ import annotations

import asyncio
from typing import Any

from src.domain.risk.stop_win_target import resolve_stop_win_target


_LINEAR_RESET_YIELD_SECONDS = 0.1


def linear_reset_occurred(risk_manager: Any) -> bool:
    """True quando o ultimo cluster encerrou com reset linear."""
    return bool(getattr(risk_manager, "_linear_reset_occurred", False))


def consume_linear_reset_flag(orch: Any) -> bool:
    """Consome e limpa flag de reset linear no risk manager."""
    rm = orch.risk_manager
    flag = linear_reset_occurred(rm)
    rm._linear_reset_occurred = False
    return flag


def session_persistence_write_active(orch: Any) -> bool:
    """True enquanto uma escrita de persistencia de sessao esta em andamento."""
    return getattr(orch, "_session_persistence_write_active", False) is True


def session_persistence_blocks_trading_cycle(orch: Any) -> bool:
    """True quando o ciclo deve aguardar liberacao da barreira de persistencia."""
    return session_persistence_write_active(orch)


def _finalize_linear_reset_risk_state(orch: Any) -> None:
    """Garante limpeza sequencial das variaveis de recovery no RiskManager."""
    rm = orch.risk_manager
    rm.consecutive_losses_linear = 0
    rm.last_loss_stake = 0.0
    pending = getattr(rm, "pending_loss", None)
    if isinstance(pending, dict):
        for sym in list(pending.keys()):
            if float(pending.get(sym, 0.0)) <= 0.0:
                pending.pop(sym, None)


def _persist_session_state_snapshot(orch: Any) -> None:
    """Persiste metricas de sessao em disco via StateManager."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return
    risk_manager = orch.risk_manager
    target = resolve_stop_win_target(
        orch.config.get("risk_management", {}),
        float(risk_manager.initial_bankroll),
    )
    if hasattr(state_mgr, "mirror_balance"):
        state_mgr.mirror_balance(float(orch.state.balance))
    else:
        state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(risk_manager.initial_bankroll)
    if state_mgr.state.daily_stop_win_target <= 0.0:
        state_mgr.state.daily_stop_win_target = float(target)
    state_mgr.check_session_limits()
    state_mgr.save_state()


async def run_linear_reset_persistence_barrier(orch: Any) -> None:
    """Sequencia limpeza de risco, persistencia de sessao e yield pos-reset linear."""
    orch._session_persistence_write_active = True
    try:
        _finalize_linear_reset_risk_state(orch)
        await asyncio.sleep(0)
        _persist_session_state_snapshot(orch)
        await orch._persist_full_state_unlocked()
    finally:
        orch._session_persistence_write_active = False
    await asyncio.sleep(_LINEAR_RESET_YIELD_SECONDS)
