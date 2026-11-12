"""Contexto de correlacao de ciclo para o formatador Aether."""

from __future__ import annotations

from contextvars import ContextVar

from src.domain.symbols.drift_symbols import symbol_display_name
from src.infrastructure.telemetry.otel_tracer import current_trace_id


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
    """Retorna prefixo `[cN|NOME]` (ou com `t=` quando ha trace ativo) ou vazio."""
    cid = _cycle_id.get()
    sym = _symbol.get()
    tid = current_trace_id()
    trace_tag = f"|t={tid[:8]}" if tid else ""
    if cid is None and not sym:
        return f"[t={tid[:8]}] " if tid else ""
    label = symbol_display_name(sym) if sym else None
    if cid is not None and label:
        return f"[c{int(cid)}|{label}{trace_tag}] "
    if cid is not None:
        return f"[c{int(cid)}{trace_tag}] "
    return f"[{label}{trace_tag}] "


def get_log_context() -> tuple[int | None, str | None]:
    """Retorna (cycle_id, symbol) associados ao contexto atual."""
    return _cycle_id.get(), _symbol.get()


def restore_log_context(cycle_id: int | None, symbol: str | None) -> None:
    """Restaura cycle_id e symbol ao contexto."""
    _cycle_id.set(int(cycle_id) if cycle_id is not None else None)
    _symbol.set(str(symbol) if symbol is not None else None)
