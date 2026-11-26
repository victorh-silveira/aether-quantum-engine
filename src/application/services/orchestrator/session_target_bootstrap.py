"""Bootstrap e persistencia da banca inicial da sessao Laya."""

from __future__ import annotations

import contextlib
from typing import Any


REDIS_SESSION_START_BALANCE_KEY = "session:start_balance"


async def bootstrap_active_session_targets(orch: Any, live_balance: float) -> None:
    """Captura banca inicial da sessao para rastreamento continuo."""
    if getattr(orch, "_session_targets_bootstrapped", False):
        return
    start_balance = max(0.0, float(live_balance))
    if hasattr(orch, "state_mgr") and orch.state_mgr is not None:
        orch.state_mgr.reset_session_metrics(start_balance, 0.0)
    orch.risk_manager.reset_session(start_balance)
    orch._session_targets_bootstrapped = True
    bm = getattr(orch, "business_metrics", None)
    if bm is not None:
        bm.set_session_targets(start_balance, 0.0)
        bm.update_balance(start_balance)
    orch.logger.info("SESSAO | Banca inicial=$%.2f | Modo continuo 24/7 (zero SKIP)", start_balance)
    store = getattr(orch, "state_store", None)
    if store is not None and hasattr(store, "set_string"):
        with contextlib.suppress(Exception):
            await store.set_string(REDIS_SESSION_START_BALANCE_KEY, str(start_balance))


async def restore_current_session_targets(orch: Any) -> None:
    """Restaura banca inicial da sessao viva apos reconnect."""
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


async def clear_current_session_redis_keys(orch: Any) -> None:
    """Limpa chaves de sessao no shutdown gracioso."""
    store = getattr(orch, "state_store", None)
    if store is None or not hasattr(store, "delete_key"):
        return
    with contextlib.suppress(Exception):
        await store.delete_key(REDIS_SESSION_START_BALANCE_KEY)
