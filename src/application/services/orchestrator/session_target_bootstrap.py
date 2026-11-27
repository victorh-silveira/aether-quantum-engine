"""Bootstrap e persistencia da banca inicial da sessao Laya."""

from __future__ import annotations

import contextlib
from typing import Any


REDIS_SESSION_START_BALANCE_KEY = "session:start_balance"
REDIS_SESSION_TARGET_WIN_KEY = "session:target_win"


async def bootstrap_active_session_targets(orch: Any, live_balance: float) -> None:
    """Captura banca inicial e meta de stop-win da sessao."""
    if getattr(orch, "_session_targets_bootstrapped", False):
        return
    start_balance = max(0.0, float(live_balance))
    params = (getattr(orch, "config", None) or {}).get("risk_management", {}).get("params", {})
    stop_win_pct = max(0.0, float(params.get("stop_win_pct", 0.0) or 0.0))
    target_win = round(start_balance * stop_win_pct, 2) if stop_win_pct > 0.0 else 0.0
    if hasattr(orch, "state_mgr") and orch.state_mgr is not None:
        orch.state_mgr.reset_session_metrics(start_balance, target_win)
    orch.risk_manager.reset_session(start_balance)
    orch._session_targets_bootstrapped = True
    bm = getattr(orch, "business_metrics", None)
    if bm is not None:
        bm.set_session_targets(start_balance, target_win)
        bm.update_balance(start_balance)
    if target_win > 0.0:
        orch.logger.info(
            "SESSAO | Banca inicial=$%.2f | Stop-win=+%.2f%% ($%.2f -> meta=$%.2f)",
            start_balance,
            stop_win_pct * 100.0,
            target_win,
            start_balance + target_win,
        )
    else:
        orch.logger.info("SESSAO | Banca inicial=$%.2f | Modo continuo 24/7 (zero SKIP)", start_balance)
    store = getattr(orch, "state_store", None)
    if store is not None and hasattr(store, "set_string"):
        with contextlib.suppress(Exception):
            await store.set_string(REDIS_SESSION_START_BALANCE_KEY, str(start_balance))
            await store.set_string(REDIS_SESSION_TARGET_WIN_KEY, str(target_win))


async def restore_current_session_targets(orch: Any) -> None:
    """Restaura banca inicial e meta de stop-win da sessao viva apos reconnect."""
    if not getattr(orch, "_session_targets_bootstrapped", False):
        return
    store = getattr(orch, "state_store", None)
    if store is None or not hasattr(orch, "state_mgr"):
        return
    start_raw = await store.get_string(REDIS_SESSION_START_BALANCE_KEY)
    if not start_raw:
        return
    try:
        start_balance = float(start_raw)
    except (ValueError, TypeError):
        return
    orch.state_mgr.state.initial_balance = start_balance
    orch.risk_manager.initial_bankroll = start_balance
    target_raw = await store.get_string(REDIS_SESSION_TARGET_WIN_KEY)
    if target_raw:
        with contextlib.suppress(ValueError, TypeError):
            orch.state_mgr.state.daily_stop_win_target = float(target_raw)


async def clear_current_session_redis_keys(orch: Any) -> None:
    """Limpa chaves de sessao no shutdown gracioso."""
    store = getattr(orch, "state_store", None)
    if store is None or not hasattr(store, "delete_key"):
        return
    with contextlib.suppress(Exception):
        await store.delete_key(REDIS_SESSION_START_BALANCE_KEY)
        await store.delete_key(REDIS_SESSION_TARGET_WIN_KEY)
