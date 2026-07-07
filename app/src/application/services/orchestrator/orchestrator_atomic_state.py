"""Contexto atomico compartilhado para leitura e escrita de estado de sessao e risco."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def orchestrator_atomic_state_context(orch: Any) -> AsyncIterator[None]:
    """Serializa mutacoes de estado de sessao, risco e persistencia no orquestrador."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        async with state_mgr.atomic_state_context():
            yield
    else:
        yield


def orchestrator_balance_snapshot(orch: Any) -> float:
    """Retorna saldo cacheado para leituras de infraestrutura fora do lock principal."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and hasattr(state_mgr, "read_cached_balance"):
        cached = float(state_mgr.read_cached_balance())
        if cached > 0.0:
            return cached
    trading_state = getattr(orch, "state", None)
    return float(getattr(trading_state, "balance", 0.0))
