"""Restore de sessao ativa e assinaturas Redis."""

from __future__ import annotations

import contextlib
from typing import Any


async def restore_session_hash(orch: Any, store: Any) -> None:
    """Restaura SessionState a partir do hash session:current."""
    session_hash = await store.get_hash("session:current")
    if not session_hash:
        session_hash = await store.get_hash("session:daily")
    if not session_hash or not hasattr(orch, "state_mgr"):
        return
    mgr = orch.state_mgr.state
    if "initial_balance" in session_hash:
        mgr.initial_balance = float(session_hash["initial_balance"])
    if "current_balance" in session_hash:
        mgr.current_balance = float(session_hash["current_balance"])
    if "daily_stop_win_target" in session_hash:
        mgr.daily_stop_win_target = float(session_hash["daily_stop_win_target"])
    if "total_trades_today" in session_hash:
        mgr.total_trades_today = int(session_hash["total_trades_today"])
    if "stop_win_triggered" in session_hash:
        mgr.stop_win_triggered = session_hash["stop_win_triggered"].lower() == "true"
    risk_manager = getattr(orch, "risk_manager", None)
    if risk_manager is not None:
        if isinstance(mgr.initial_balance, (int, float)) and float(mgr.initial_balance) > 0.0:
            risk_manager.initial_bankroll = float(mgr.initial_balance)
        if isinstance(mgr.daily_stop_win_target, (int, float)) and float(mgr.daily_stop_win_target) > 0.0:
            risk_manager.daily_stop_win_target = float(mgr.daily_stop_win_target)


async def restore_market_signatures(orch: Any, store: Any) -> None:
    """Restaura assinaturas de mercado e ultima barra processada."""
    market_sig = await store.get_string("market_sig")
    if market_sig:
        orch.last_data_signature = market_sig
    anchor = getattr(orch, "anchor", None)
    if not anchor:
        return
    bar_sig = await store.get_string(f"bar_sig:{anchor}")
    if not bar_sig:
        return
    with contextlib.suppress(ValueError):
        orch._last_processed_epoch = int(bar_sig)
