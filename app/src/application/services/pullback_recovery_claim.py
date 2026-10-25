"""Reserva unica da recuperacao por conta entre ordens simultaneas."""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from src.infrastructure.state.redis_state_store import RedisStateStore


logger = logging.getLogger("AETH")


def _account_key(orch) -> str:
    """Nomeia a reserva pela conta e pelo ambiente de negociacao."""
    mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
    account = str(getattr(orch, "deriv_account_id", "") or "default")
    return f"pullback_recovery:{mode}:{account}"


def _lock(orch) -> asyncio.Lock:
    """Devolve o lock local da reserva da conta."""
    lock = getattr(orch, "_pullback_recovery_lock", None)
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        orch._pullback_recovery_lock = lock
    return lock


def _claims(orch) -> dict[int | str, tuple[str, float, float]]:
    """Devolve as parcelas da perda ja atribuidas a ordens."""
    claims = getattr(orch, "_pullback_recovery_claims", None)
    if not isinstance(claims, dict):
        claims = {}
        orch._pullback_recovery_claims = claims
    now = time.monotonic()
    for identity, (_, _, expires_at) in list(claims.items()):
        if expires_at <= now:
            claims.pop(identity, None)
    return claims


def unreserved_loss(orch) -> float:
    """Desconta a perda ja atribuida a contratos abertos."""
    pending = max(0.0, float(orch.risk_manager.pending_loss_total()))
    claims = _claims(orch)
    if pending > 0 and getattr(orch.risk_manager, "active_contract_ids", None) and not claims:
        return 0.0
    return max(0.0, pending - sum(amount for _, amount, _ in claims.values()))


async def claim_recovery(orch, amount: float) -> str | None:
    """Reserva uma parcela da perda localmente e no Redis ate a liquidacao."""
    if amount <= 0:
        return None
    async with _lock(orch):
        if _claims(orch):
            return None
        token = uuid4().hex
        store = getattr(orch, "state_store", None)
        if isinstance(store, RedisStateStore):
            try:
                client = await store._redis()
                if not await client.set(store._full_key(_account_key(orch)), token, nx=True, ex=900):
                    return None
            except Exception:
                return None
        _claims(orch)[token] = (token, amount, time.monotonic() + 900)
        return token


def bind_recovery_contract(orch, token: str, contract_id: int) -> None:
    """Mantem a reserva ate o contrato comprado ser liquidado."""
    claim = _claims(orch).pop(token, None)
    if claim is not None:
        tok, amt, exp = claim
        _claims(orch)[int(contract_id)] = (tok, amt, exp + 900)


async def release_recovery(orch, identity: int | str) -> None:
    """Libera apenas a propria reserva, sem apagar uma compra concorrente."""
    async with _lock(orch):
        claim = _claims(orch).pop(identity, None)
        if claim is None:
            return
        token, _, _ = claim
        store = getattr(orch, "state_store", None)
        if isinstance(store, RedisStateStore):
            script = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
            try:
                client = await store._redis()
                await client.eval(script, 1, store._full_key(_account_key(orch)), token)
            except Exception as exc:
                logger.warning("RECOVERY: falha ao liberar reserva Redis: %s", exc)
