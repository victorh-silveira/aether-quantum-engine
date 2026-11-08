"""Restore de estado Redis no boot do orquestrador."""

from __future__ import annotations

import contextlib
from typing import Any

from src.application.services.orchestrator.orchestrator_state_session import (
    restore_market_signatures,
    restore_session_hash,
)
from src.application.services.orchestrator.session_target_bootstrap import restore_current_session_targets
from src.infrastructure.state.redis_ephemeral_ttl import REDIS_EPHEMERAL_SIG_TTL_SECONDS


async def restore_orchestrator_state(orch: Any) -> None:
    """Carrega snapshot Redis e restaura RiskManager e assinaturas."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return
    snapshot = await store.load_snapshot()
    if isinstance(snapshot, dict):
        risk = snapshot.get("risk")
        pending_hash = await store.get_hash("state:pending_loss")
        if pending_hash:
            merged = dict(risk) if isinstance(risk, dict) else {}
            merged["pending_loss"] = {str(k): float(v) for k, v in pending_hash.items()}
            risk = merged
        if isinstance(risk, dict) and hasattr(orch.risk_manager, "restore_state"):
            orch.risk_manager.restore_state(risk)
        profit = snapshot.get("total_session_profit")
        if profit is not None and not getattr(orch, "_session_targets_bootstrapped", False):
            orch.risk_manager.total_session_profit = float(profit)
    await restore_session_hash(orch, store)
    await restore_current_session_targets(orch)
    await restore_market_signatures(orch, store)


def session_hash_payload(orch: Any) -> dict[str, float | int | bool]:
    """Serializa metricas da sessao ativa para hash Redis."""
    mgr = orch.state_mgr.state
    return {
        "initial_balance": mgr.initial_balance,
        "current_balance": mgr.current_balance,
        "daily_stop_win_target": mgr.daily_stop_win_target,
        "total_trades_today": mgr.total_trades_today,
        "stop_win_triggered": mgr.stop_win_triggered,
    }


async def persist_session_hash(orch: Any) -> None:
    """Grava hash session:current com metricas correntes."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return
    await store.set_hash("session:current", session_hash_payload(orch))


async def bar_epoch_already_processed(orch: Any, symbol: str, epoch: int) -> bool:
    """True quando epoch de barra ja foi processado nesta sessao."""
    store = getattr(orch, "state_store", None)
    infra = getattr(orch, "infra", None)
    if store is None or infra is None or not infra.enabled:
        return False
    stored = await store.get_string(f"bar_sig:{symbol}")
    if not stored:
        return False
    with contextlib.suppress(ValueError):
        return int(stored) == int(epoch)
    return False


async def mark_bar_processed(orch: Any, symbol: str, epoch: int) -> None:
    """Marca epoch de barra como processado no Redis."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return
    await store.set_string(
        f"bar_sig:{symbol}",
        str(int(epoch)),
        ttl_seconds=REDIS_EPHEMERAL_SIG_TTL_SECONDS,
    )


async def sync_market_signature(orch: Any, signature: str) -> None:
    """Persiste assinatura agregada de OHLC para deduplicacao de ciclo."""
    store = getattr(orch, "state_store", None)
    if store is None or not signature:
        return
    await store.set_string(
        "market_sig",
        signature,
        ttl_seconds=REDIS_EPHEMERAL_SIG_TTL_SECONDS,
    )
