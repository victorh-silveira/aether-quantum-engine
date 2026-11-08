"""Cache de predicao DL por fronteira temporal, cycle_id e fingerprint de tensor."""

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
    cycle_id: int = 0,
) -> None:
    """Persiste entrada de decisao indexada por simbolo, ciclo e fingerprint."""
    prediction_cache(orch)[str(symbol)] = {
        "entry": entry,
        "tensor_fingerprint": tensor_fingerprint,
        "boundary_epoch": int(boundary_epoch),
        "cycle_id": int(cycle_id),
    }


def resolve_cached_prediction(
    orch: Any,
    symbol: str,
    *,
    at_boundary: bool,
    tensor_fingerprint: bytes | None = None,
    boundary_epoch: int | None = None,
    cycle_id: int | None = None,
) -> dict[str, Any] | None:
    """Retorna predicao em cache no mesmo cycle_id e boundary_epoch."""
    slot = prediction_cache(orch).get(str(symbol))
    if slot is None:
        return None
    entry = slot.get("entry")
    if not isinstance(entry, dict):
        return None
    if cycle_id is not None and int(cycle_id) > 0:
        try:
            if int(slot.get("cycle_id", -1)) != int(cycle_id):
                return None
        except (TypeError, ValueError):
            return None
    if boundary_epoch is not None:
        try:
            if int(slot.get("boundary_epoch", -1)) != int(boundary_epoch):
                return None
        except (TypeError, ValueError):
            return None
    if tensor_fingerprint is not None:
        if slot.get("tensor_fingerprint") == tensor_fingerprint:
            return entry
        return None
    if not at_boundary:
        return entry
    return None
