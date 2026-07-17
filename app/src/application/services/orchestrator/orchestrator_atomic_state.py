"""Contexto atomico compartilhado para leitura e escrita de estado de sessao e risco."""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


logger = logging.getLogger("AETH")
_STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS = 5.0


def _caller_function_name() -> str:
    """Resolve o nome da funcao externa que pediu o lock atomico."""
    frame = inspect.currentframe()
    if frame is None:
        return "<unknown>"
    outer = frame.f_back
    while outer is not None:
        module = outer.f_globals.get("__name__", "")
        if module not in {"contextlib", __name__} and not str(module).startswith("contextlib"):
            return outer.f_code.co_name
        outer = outer.f_back
    return "<unknown>"


@asynccontextmanager
async def orchestrator_atomic_state_context(orch: Any) -> AsyncIterator[None]:
    """Serializa mutacoes de estado de sessao, risco e persistencia no orquestrador."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        lock = state_mgr._state_lock
        caller = inspect.stack()[1].function
        logger.debug(f"[LOCK_TRACE] Tentando adquirir _state_lock invocado por: {caller}")
        try:
            await asyncio.wait_for(lock.acquire(), timeout=_STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            holder = "ocupado" if lock.locked() else "indefinido"
            origin = _caller_function_name()
            stack = "".join(traceback.format_stack())
            raise RuntimeError(
                f"[AETHER] STATE_LOCK_TIMEOUT: deadlock em _state_lock "
                f"(caller={origin}, retentor={holder}, timeout={_STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS:.1f}s)\n"
                f"{stack}"
            ) from exc
        try:
            yield
        finally:
            lock.release()
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
