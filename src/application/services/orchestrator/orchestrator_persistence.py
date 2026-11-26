"""Persistencia atomica de snapshot de sessao, risco e assinaturas de mercado."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.orchestrator_data_signature import get_data_state_signature
from src.application.services.orchestrator.orchestrator_state_restore import (
    persist_session_hash,
    session_hash_payload,
    sync_market_signature,
)


logger = logging.getLogger("AETH")


async def persist_full_state_unlocked(orch: Any) -> None:
    """Persiste snapshot completo assumindo lock atomico ja adquirido."""
    s = await orch.state.get_state()
    s.update(
        {
            "total_session_profit": orch.risk_manager.total_session_profit,
            "risk": orch.risk_manager.get_state(),
        }
    )
    sig = get_data_state_signature(orch)
    session = session_hash_payload(orch)
    start_bal = float(getattr(orch.risk_manager, "initial_bankroll", 0.0) or 0.0)
    save_bundle = getattr(orch.state_store, "save_state_bundle", None)
    if callable(save_bundle):
        await save_bundle(
            snapshot=s,
            session=session,
            market_sig=sig or None,
            session_start_balance=start_bal,
        )
    else:
        await orch.state_store.save_snapshot(s)
        await persist_session_hash(orch)
        if sig:
            await sync_market_signature(orch, sig)
    if hasattr(orch.persistence, "save"):
        orch.persistence.save(s)
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and hasattr(state_mgr, "mirror_balance"):
        state_mgr.mirror_balance(float(orch.state.balance))


async def save_full_state(orch: Any, *, raise_on_timeout: bool = False) -> bool:
    """Persiste snapshot completo sob lock atomico; nao derruba o motor se o lock estiver ocupado."""
    try:
        async with orchestrator_atomic_state_context(orch):
            await persist_full_state_unlocked(orch)
        return True
    except RuntimeError as exc:
        if "STATE_LOCK_TIMEOUT" not in str(exc):
            raise
        logger.warning("STATE_LOCK_BUSY | save_full_state adiado (lock ocupado)")
        if raise_on_timeout:
            raise
        return False
