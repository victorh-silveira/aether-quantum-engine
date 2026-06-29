"""StateStore Redis com debounce de snapshot."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis


class RedisStateStore:
    """Persistencia de estado e assinaturas em Redis."""

    def __init__(self, *, url: str, key_prefix: str = "aether", debounce_seconds: float = 1.0):
        self._url = url
        self._prefix = key_prefix.rstrip(":")
        self._debounce = max(0.0, float(debounce_seconds))
        self._client: aioredis.Redis | None = None
        self._last_snapshot_at = 0.0
        self._pending_snapshot: dict[str, Any] | None = None
        self.logger = logging.getLogger("AETH")

    def _full_key(self, suffix: str) -> str:
        """Monta chave Redis com prefixo configurado."""
        return f"{self._prefix}:{suffix}"

    async def _redis(self) -> aioredis.Redis:
        """Retorna cliente Redis lazy singleton."""
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def save_snapshot(self, payload: dict[str, Any]) -> None:
        """Persiste snapshot com debounce opcional."""
        now = time.monotonic()
        if self._debounce > 0 and (now - self._last_snapshot_at) < self._debounce:
            self._pending_snapshot = payload
            return
        await self._write_snapshot(payload)
        self._last_snapshot_at = now
        self._pending_snapshot = None

    async def _write_snapshot(self, payload: dict[str, Any]) -> None:
        """Grava snapshot JSON e hash de risco no Redis."""
        client = await self._redis()
        key = self._full_key("state:snapshot")
        risk = payload.get("risk")
        if isinstance(risk, dict):
            flat = {str(k): str(v) for k, v in risk.items() if not isinstance(v, (dict, list))}
            if flat:
                await client.hset(self._full_key("state:risk"), mapping=flat)
        await client.set(key, json.dumps(payload))

    async def flush_snapshot(self) -> None:
        """Forca gravacao de snapshot pendente por debounce."""
        if self._pending_snapshot is not None:
            await self._write_snapshot(self._pending_snapshot)
            self._last_snapshot_at = time.monotonic()
            self._pending_snapshot = None

    async def load_snapshot(self) -> dict[str, Any] | None:
        """Carrega snapshot JSON do Redis."""
        client = await self._redis()
        raw = await client.get(self._full_key("state:snapshot"))
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    async def set_hash(self, key: str, mapping: dict[str, Any]) -> None:
        """Grava hash Redis com valores stringificados."""
        client = await self._redis()
        flat = {str(k): str(v) for k, v in mapping.items()}
        if flat:
            await client.hset(self._full_key(key), mapping=flat)

    async def get_hash(self, key: str) -> dict[str, str]:
        """Le hash Redis completo."""
        client = await self._redis()
        data = await client.hgetall(self._full_key(key))
        return data if isinstance(data, dict) else {}

    async def set_string(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        """Grava string Redis com TTL opcional."""
        client = await self._redis()
        full = self._full_key(key)
        if ttl_seconds is not None and int(ttl_seconds) > 0:
            await client.setex(full, int(ttl_seconds), str(value))
        else:
            await client.set(full, str(value))

    async def get_string(self, key: str) -> str | None:
        """Le string Redis por chave relativa."""
        client = await self._redis()
        return await client.get(self._full_key(key))

    async def ping(self) -> bool:
        """Valida conectividade Redis com PING."""
        client = await self._redis()
        return (await client.ping()) is True

    async def close(self) -> None:
        """Flush final e encerra conexao Redis."""
        await self.flush_snapshot()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
