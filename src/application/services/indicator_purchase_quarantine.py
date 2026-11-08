"""Quarentena de compras cujo ACK nao chegou do broker."""

import logging
import time

from src.infrastructure.state.redis_state_store import RedisStateStore


logger = logging.getLogger("AETH")
UNKNOWN_PURCHASE_TTL_SECONDS = 900


def purchase_account_key(orch, symbol: str) -> str:
    """Identifica conta e ativo sem misturar DEMO e REAL."""
    mode = str(orch.config.get("trading", {}).get("mode", "demo"))
    account = str(getattr(orch, "deriv_account_id", "") or "default")
    return f"{mode}:{account}:{symbol}"


async def quarantine_unknown_purchase(orch, symbol: str) -> None:
    """Suspende novas ordens enquanto uma compra sem ACK pode estar aberta."""
    until = time.time() + UNKNOWN_PURCHASE_TTL_SECONDS
    blocked = getattr(orch, "_unknown_purchase_until", None)
    if not isinstance(blocked, dict):
        blocked = {}
        orch._unknown_purchase_until = blocked
    blocked[str(symbol)] = until
    store = getattr(orch, "state_store", None)
    if isinstance(store, RedisStateStore):
        try:
            client = await store._redis()
            await client.set(
                store._full_key(f"unknown_purchase:{purchase_account_key(orch, symbol)}"),
                "unknown",
                ex=UNKNOWN_PURCHASE_TTL_SECONDS,
            )
        except Exception as exc:
            logger.error("EXEC: falha ao persistir compra sem ACK para %s: %s", symbol, exc)
