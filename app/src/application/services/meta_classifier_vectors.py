"""Store de vetores pre-trade do meta-regressor (symbol + contract_id)."""

from __future__ import annotations

from typing import Any


def _store(orch: Any) -> dict[str, list[float]]:
    """Retorna o dict de vetores meta no orch, criando se necessario."""
    store = getattr(orch, "_meta_clf_vectors", None)
    if not isinstance(store, dict):
        store = {}
        orch._meta_clf_vectors = store
    return store


def store_meta_feature_vector(orch: Any, symbol: str, vector: list[float]) -> None:
    """Guarda vetor meta do ultimo prefetch por simbolo."""
    if not symbol or not isinstance(vector, list) or not vector:
        return
    _store(orch)[str(symbol)] = list(vector)


def bind_meta_feature_vector_to_contract(orch: Any, symbol: str, contract_id: int) -> None:
    """Copia vetor do simbolo para chave cid apos EXEC confirmado."""
    store = _store(orch)
    vector = store.get(str(symbol))
    if not isinstance(vector, list) or not vector:
        return
    store[f"cid:{int(contract_id)}"] = list(vector)


def pop_meta_feature_vector(orch: Any, symbol: str, contract_id: int) -> list[float] | None:
    """Prefere cid; fallback symbol; remove ambas chaves usadas."""
    store = getattr(orch, "_meta_clf_vectors", None)
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
