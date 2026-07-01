"""Bootstrap e persistencia de metas de stop win por sessao ativa."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stop_win_target import (
    REDIS_SESSION_START_BALANCE_KEY,
    REDIS_SESSION_TARGET_WIN_KEY,
    StopWinManager,
    resolve_session_start_balance,
)


async def bootstrap_active_session_targets(orch: Any, live_balance: float) -> None:
    """Captura banca inicial da sessao e define meta de 1% composto."""
    if getattr(orch, "_session_targets_bootstrapped", False):
        return
    risk_cfg = orch.config.get("risk_management", {}) if isinstance(getattr(orch, "config", {}), dict) else {}
    start_balance = resolve_session_start_balance(live_balance, risk_cfg)
    swm = StopWinManager(risk_cfg)
    if swm.is_compounding_enabled():
        target_win = swm.calculate_session_targets(start_balance).target_win
    else:
        target_win = swm.resolve_target(start_balance)
    orch.state_mgr.reset_session_metrics(start_balance, target_win)
    orch.risk_manager.reset_session(start_balance, target=target_win)
    orch.risk_manager.total_session_profit = 0.0
    orch._session_targets_bootstrapped = True
    orch.logger.info(
        "SESSAO INICIADA | Alvo de 1%%: $%.2f | Stop Loss: DESATIVADO",
        target_win,
    )
    await _persist_current_session_keys(orch, start_balance, target_win)


async def restore_current_session_targets(orch: Any) -> None:
    """Restaura metas da sessao viva apos reconnect dentro do mesmo processo."""
    if not getattr(orch, "_session_targets_bootstrapped", False):
        return
    store = getattr(orch, "state_store", None)
    if store is None or not hasattr(orch, "state_mgr"):
        return
    start_raw = await store.get_string(REDIS_SESSION_START_BALANCE_KEY)
    target_raw = await store.get_string(REDIS_SESSION_TARGET_WIN_KEY)
    if not start_raw or not target_raw:
        return
    try:
        start_balance = float(start_raw)
        target_win = float(target_raw)
    except (ValueError, TypeError):
        return
    mgr = orch.state_mgr.state
    mgr.initial_balance = start_balance
    mgr.daily_stop_win_target = target_win
    if hasattr(orch, "risk_manager"):
        orch.risk_manager.initial_bankroll = start_balance
        orch.risk_manager.daily_stop_win_target = target_win


async def clear_current_session_redis_keys(orch: Any) -> None:
    """Remove chaves de sessao corrente no encerramento gracioso."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return
    await store.delete_string(REDIS_SESSION_START_BALANCE_KEY)
    await store.delete_string(REDIS_SESSION_TARGET_WIN_KEY)


def current_session_redis_payload(orch: Any) -> tuple[float | None, float | None]:
    """Retorna banca inicial e meta win da sessao ativa para persistencia atomica."""
    if not getattr(orch, "_session_targets_bootstrapped", False):
        return None, None
    mgr = orch.state_mgr.state
    start = getattr(mgr, "initial_balance", None)
    target = getattr(mgr, "daily_stop_win_target", None)
    if not isinstance(start, (int, float)) or not isinstance(target, (int, float)):
        return None, None
    if float(start) <= 0.0 or float(target) <= 0.0:
        return None, None
    return float(start), float(target)


async def _persist_current_session_keys(orch: Any, start_balance: float, target_win: float) -> None:
    """Grava chaves session:current:* no state store."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return
    await store.set_string(REDIS_SESSION_START_BALANCE_KEY, str(float(start_balance)))
    await store.set_string(REDIS_SESSION_TARGET_WIN_KEY, str(float(target_win)))
