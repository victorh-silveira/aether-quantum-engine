"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

from typing import Any

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


async def acquire_trading_cycle_lock(orch: Any) -> bool:
    """Reserva o slot de ciclo ativo; False se outro ciclo ja esta em andamento."""
    async with orch.lock:
        if _stop_win_blocks_cycle(orch):
            return False
        if orch.is_trading:
            return False
        orch.is_trading = True
    return True
