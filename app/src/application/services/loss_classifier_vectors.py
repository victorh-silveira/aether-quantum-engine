"""Store de vetores pre-trade do loss-classifier (symbol + contract_id)."""

from __future__ import annotations

from typing import Any


def _store(orch: Any) -> dict[str, list[float]]:
    """Retorna o dict de vetores loss-clf no orch, criando se necessario."""
    store = getattr(orch, "_loss_clf_vectors", None)
    if not isinstance(store, dict):
        store = {}
        orch._loss_clf_vectors = store
    return store


def store_loss_feature_vector(orch: Any, symbol: str, vector: list[float]) -> None:
    """Guarda vetor do ultimo gate por simbolo."""
    if not symbol or not isinstance(vector, list) or not vector:
        return
    _store(orch)[str(symbol)] = list(vector)


def bind_loss_feature_vector_to_contract(orch: Any, symbol: str, contract_id: int) -> None:
    """Copia vetor do simbolo para chave cid apos EXEC confirmado."""
    store = _store(orch)
    vector = store.get(str(symbol))
    if not isinstance(vector, list) or not vector:
        return
    store[f"cid:{int(contract_id)}"] = list(vector)


def store_loss_flip_ctx(orch: Any, symbol: str, ctx: dict[str, Any]) -> None:
    """Guarda telemetria de FLIP do ultimo gate por simbolo."""
    if not symbol or not isinstance(ctx, dict):
        return
    store = getattr(orch, "_loss_clf_flip_ctx", None)
    if not isinstance(store, dict):
        store = {}
        orch._loss_clf_flip_ctx = store
    store[str(symbol)] = dict(ctx)


def bind_loss_flip_ctx_to_contract(orch: Any, symbol: str, contract_id: int) -> None:
    """Copia ctx de FLIP do simbolo para chave cid apos EXEC."""
    store = getattr(orch, "_loss_clf_flip_ctx", None)
    if not isinstance(store, dict):
        return
    ctx = store.get(str(symbol))
    if not isinstance(ctx, dict):
        return
    store[f"cid:{int(contract_id)}"] = dict(ctx)


def pop_loss_flip_ctx(orch: Any, symbol: str, contract_id: int) -> dict[str, Any] | None:
    """Prefere cid; fallback symbol."""
    store = getattr(orch, "_loss_clf_flip_ctx", None)
    if not isinstance(store, dict):
        return None
    cid_key = f"cid:{int(contract_id)}"
    ctx = store.pop(cid_key, None)
    if isinstance(ctx, dict):
        store.pop(str(symbol), None)
        return dict(ctx)
    fallback = store.pop(str(symbol), None)
    if isinstance(fallback, dict):
        return dict(fallback)
    return None


def pop_loss_feature_vector(orch: Any, symbol: str, contract_id: int) -> list[float] | None:
    """Prefere cid; fallback symbol; remove ambas chaves usadas."""
    store = getattr(orch, "_loss_clf_vectors", None)
    if not isinstance(store, dict):
        return None
    cid_key = f"cid:{int(contract_id)}"
    vector = store.pop(cid_key, None)
    if isinstance(vector, list) and vector:
        store.pop(str(symbol), None)
        return list(vector)
    fallback = store.pop(str(symbol), None)
    if isinstance(fallback, list) and fallback:
        return list(fallback)
    return None
