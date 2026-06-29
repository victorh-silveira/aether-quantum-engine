"""Escrita atomica de snapshot e hashes no Redis via pipeline."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis


def _flat_scalars(data: dict[str, Any]) -> dict[str, str]:
    """Achata dict de risco mantendo apenas valores escalares."""
    return {str(k): str(v) for k, v in data.items() if not isinstance(v, (dict, list))}


def _flat_mapping(data: dict[str, Any] | None) -> dict[str, str]:
    """Converte dict arbitrario em mapeamento string para HSET."""
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


async def write_state_bundle(
    client: aioredis.Redis,
    *,
    prefix: str,
    snapshot: dict[str, Any],
    session_hash: dict[str, Any] | None = None,
    market_sig: str | None = None,
) -> None:
    """Grava snapshot, risco, pending_loss, sessao e assinatura em transacao."""
    pfx = prefix.rstrip(":")
    snapshot_key = f"{pfx}:state:snapshot"
    risk_key = f"{pfx}:state:risk"
    pending_key = f"{pfx}:state:pending_loss"
    session_key = f"{pfx}:session:daily"
    market_key = f"{pfx}:market_sig"
    risk = snapshot.get("risk")
    async with client.pipeline(transaction=True) as pipe:
        pipe.set(snapshot_key, json.dumps(snapshot))
        if isinstance(risk, dict):
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
        session_flat = _flat_mapping(session_hash)
        if session_flat:
            pipe.hset(session_key, mapping=session_flat)
        if market_sig:
            pipe.set(market_key, str(market_sig))
        await pipe.execute()
