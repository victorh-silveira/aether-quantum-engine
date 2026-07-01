"""Escrita atomica de snapshot e hashes no Redis via pipeline MULTI/EXEC."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from src.domain.risk.recovery_hurst_decay import REDIS_SKIP_COUNTER_KEY
from src.domain.risk.stop_win_target import (
    REDIS_SESSION_START_BALANCE_KEY,
    REDIS_SESSION_TARGET_WIN_KEY,
)


def _flat_scalars(data: dict[str, Any]) -> dict[str, str]:
    """Converte dict para mapa string-string omitindo estruturas aninhadas."""
    return {str(k): str(v) for k, v in data.items() if not isinstance(v, (dict, list))}


def _flat_mapping(data: dict[str, Any] | None) -> dict[str, str]:
    """Normaliza dict opcional para mapa string-string."""
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _queue_risk_hashes(pipe: Any, pfx: str, risk: dict[str, Any] | None) -> None:
    """Enfileira hashes de risco e pending_loss no pipeline Redis."""
    risk_key = f"{pfx}:state:risk"
    pending_key = f"{pfx}:state:pending_loss"
    if not isinstance(risk, dict):
        return
    flat = _flat_scalars(risk)
    if flat:
        pipe.hset(risk_key, mapping=flat)
    pending = risk.get("pending_loss")
    if isinstance(pending, dict):
        pending_flat = _flat_mapping(pending)
        if pending_flat:
            pipe.delete(pending_key)
            pipe.hset(pending_key, mapping=pending_flat)
        else:
            pipe.delete(pending_key)


async def write_state_bundle(
    client: aioredis.Redis,
    *,
    prefix: str,
    snapshot: dict[str, Any],
    session_hash: dict[str, Any] | None = None,
    market_sig: str | None = None,
    recovery_skip_counter: int | None = None,
    session_start_balance: float | None = None,
    session_target_win: float | None = None,
) -> None:
    """Grava snapshot, risco, pending_loss, sessao, skip counter e assinatura em transacao."""
    pfx = prefix.rstrip(":")
    snapshot_key = f"{pfx}:state:snapshot"
    session_key = f"{pfx}:session:current"
    market_key = f"{pfx}:market_sig"
    skip_key = f"{pfx}:{REDIS_SKIP_COUNTER_KEY}"
    start_key = f"{pfx}:{REDIS_SESSION_START_BALANCE_KEY}"
    target_key = f"{pfx}:{REDIS_SESSION_TARGET_WIN_KEY}"
    risk = snapshot.get("risk")
    async with client.pipeline(transaction=True) as pipe:
        pipe.set(snapshot_key, json.dumps(snapshot))
        _queue_risk_hashes(pipe, pfx, risk if isinstance(risk, dict) else None)
        session_flat = _flat_mapping(session_hash)
        if session_flat:
            pipe.hset(session_key, mapping=session_flat)
        if market_sig:
            pipe.set(market_key, str(market_sig))
        if recovery_skip_counter is not None:
            pipe.set(skip_key, str(max(0, int(recovery_skip_counter))))
        if session_start_balance is not None:
            pipe.set(start_key, str(float(session_start_balance)))
        if session_target_win is not None:
            pipe.set(target_key, str(float(session_target_win)))
        await pipe.execute()
