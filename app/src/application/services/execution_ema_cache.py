"""Cache de EMA do ciclo (invalidacao no orquestrador)."""

from __future__ import annotations


_STATE: dict[str, object] = {"cycle": None, "cache": {}}


def invalidate_ema_cache(cycle_id: int | None = None) -> None:
    """Limpa o cache de EMA; chamar no inicio de cada ciclo do orquestrador."""
    cache = _STATE["cache"]
    if isinstance(cache, dict):
        cache.clear()
    _STATE["cycle"] = cycle_id
