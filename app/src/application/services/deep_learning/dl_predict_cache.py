"""Cache de predicao DL por fronteira temporal e fingerprint de tensor."""

from __future__ import annotations

from typing import Any


def prediction_cache(orch: Any) -> dict[str, dict[str, Any]]:
    """Retorna cache mutavel de predicoes por simbolo no orquestrador."""
    cache = getattr(orch, "_dl_prediction_cache", None)
    if cache is None:
        cache = {}
        orch._dl_prediction_cache = cache
    return cache


def store_prediction_cache(
    orch: Any,
    symbol: str,
    entry: dict[str, Any],
    *,
    tensor_fingerprint: bytes,
    boundary_epoch: int,
) -> None:
    """Persiste entrada de decisao indexada por simbolo e fingerprint."""
    prediction_cache(orch)[str(symbol)] = {
        "entry": entry,
        "tensor_fingerprint": tensor_fingerprint,
        "boundary_epoch": int(boundary_epoch),
    }


def resolve_cached_prediction(
    orch: Any,
    symbol: str,
    *,
    at_boundary: bool,
    tensor_fingerprint: bytes | None = None,
) -> dict[str, Any] | None:
    """Retorna predicao em cache quando fora da fronteira ou tensor duplicado."""
    slot = prediction_cache(orch).get(str(symbol))
    if slot is None:
        return None
    entry = slot.get("entry")
    if not isinstance(entry, dict):
        return None
    if not at_boundary:
        return entry
    if tensor_fingerprint is None:
        return None
    if slot.get("tensor_fingerprint") == tensor_fingerprint:
        return entry
    return None
