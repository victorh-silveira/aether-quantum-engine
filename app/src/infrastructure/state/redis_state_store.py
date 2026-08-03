"""StateStore Redis com debounce de snapshot."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from src.infrastructure.state.redis_state_pipeline import write_state_bundle


class RedisStateStore:
    """Persistencia de estado e assinaturas em Redis."""

    def __init__(
        self,
        *,
        url: str,
        key_prefix: str = "aether",
        debounce_seconds: float = 1.0,
        socket_connect_timeout: float = 2.0,
        socket_timeout: float = 15.0,
    ):
        self._url = url
        self._prefix = key_prefix.rstrip(":")
        self._debounce = max(0.0, float(debounce_seconds))
        self._socket_connect_timeout = max(0.1, float(socket_connect_timeout))
        self._socket_timeout = max(0.1, float(socket_timeout))
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
            self._client = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=self._socket_connect_timeout,
                socket_timeout=self._socket_timeout,
            )
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
        await write_state_bundle(
            client,
            prefix=self._prefix,
            snapshot=payload,
        )

    async def save_state_bundle(
        self,
        *,
        snapshot: dict[str, Any],
        session: dict[str, Any] | None = None,
        market_sig: str | None = None,
        recovery_skip_counter: int | None = None,
        session_start_balance: float | None = None,
        session_target_win: float | None = None,
        dlambert_unit: float | None = None,
        consecutive_losses_linear: int | None = None,
    ) -> None:
        """Grava bundle atomico com sessao e chaves de meta ativa."""
        client = await self._redis()
        await write_state_bundle(
            client,
            prefix=self._prefix,
            snapshot=snapshot,
            session_hash=session,
            market_sig=market_sig,
            recovery_skip_counter=recovery_skip_counter,
            session_start_balance=session_start_balance,
            session_target_win=session_target_win,
            dlambert_unit=dlambert_unit,
            consecutive_losses_linear=consecutive_losses_linear,
        )
        self._last_snapshot_at = time.monotonic()
        self._pending_snapshot = None

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
        """Le string Redis pelo sufixo de chave."""
        client = await self._redis()
        return await client.get(self._full_key(key))

    async def delete_string(self, key: str) -> None:
        """Remove string Redis pelo sufixo de chave."""
        client = await self._redis()
        await client.delete(self._full_key(key))

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
