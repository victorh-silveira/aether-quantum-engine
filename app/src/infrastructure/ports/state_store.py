"""Porta de persistencia de estado do motor."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """Contrato async para snapshot, hash e chaves de assinatura."""

    async def save_snapshot(self, payload: dict[str, Any]) -> None:
        """Persiste snapshot completo do orquestrador."""

    async def save_state_bundle(
        self,
        *,
        snapshot: dict[str, Any],
        session: dict[str, Any] | None = None,
        market_sig: str | None = None,
    ) -> None:
        """Persiste snapshot, hashes e assinatura em transacao unica."""

    async def load_snapshot(self) -> dict[str, Any] | None:
        """Carrega snapshot completo ou None."""

    async def set_hash(self, key: str, mapping: dict[str, Any]) -> None:
        """Grava hash de campos escalares."""

    async def get_hash(self, key: str) -> dict[str, str]:
        """Le hash de campos escalares."""

    async def set_string(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        """Grava valor string opcionalmente com TTL."""

    async def get_string(self, key: str) -> str | None:
        """Le valor string ou None."""

    async def ping(self) -> bool:
        """Valida conectividade."""

    async def close(self) -> None:
        """Encerra conexoes."""
