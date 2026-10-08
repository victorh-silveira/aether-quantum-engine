"""Contexto de correlacao de ciclo para o formatador Aether."""

from __future__ import annotations

from contextvars import ContextVar


_cycle_id: ContextVar[int | None] = ContextVar("aether_log_cycle_id", default=None)
_symbol: ContextVar[str | None] = ContextVar("aether_log_symbol", default=None)


def bind_log_context(*, cycle_id: int | None = None, symbol: str | None = None) -> None:
    """Associa cycle_id e/ou symbol ao contexto do task atual."""
    if cycle_id is not None:
        _cycle_id.set(int(cycle_id))
    if symbol is not None:
        _symbol.set(str(symbol))


def clear_log_context() -> None:
    """Limpa cycle_id e symbol do contexto."""
    _cycle_id.set(None)
    _symbol.set(None)


def format_log_context_prefix() -> str:
    """Retorna prefixo `[cN|SYM]` ou vazio se sem contexto."""
    cid = _cycle_id.get()
    sym = _symbol.get()
    if cid is None and not sym:
        return ""
    if cid is not None and sym:
        return f"[c{int(cid)}|{sym}] "
    if cid is not None:
        return f"[c{int(cid)}] "
    return f"[{sym}] "
