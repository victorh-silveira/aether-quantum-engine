"""Porta de persistencia de estado do motor."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """Contrato assincrono para snapshot, hashes e chaves de sessao."""

    async def save_snapshot(self, payload: dict[str, Any]) -> None:
        """Persiste snapshot JSON completo do motor."""

    async def save_state_bundle(
        self,
        *,
        snapshot: dict[str, Any],
        session: dict[str, Any] | None = None,
        market_sig: str | None = None,
        recovery_skip_counter: int | None = None,
        session_start_balance: float | None = None,
        session_target_win: float | None = None,
    ) -> None:
        """Grava snapshot, sessao e chaves auxiliares em transacao atomica."""

    async def load_snapshot(self) -> dict[str, Any] | None:
        """Carrega ultimo snapshot JSON persistido."""

    async def set_hash(self, key: str, mapping: dict[str, Any]) -> None:
        """Grava hash Redis com campos escalares."""

    async def get_hash(self, key: str) -> dict[str, str]:
        """Le hash Redis como mapa string-string."""

    async def set_string(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        """Grava chave string Redis com TTL opcional."""

    async def get_string(self, key: str) -> str | None:
        """Le valor string Redis ou None."""

    async def delete_string(self, key: str) -> None:
        """Remove chave string Redis."""

    async def ping(self) -> bool:
        """Verifica conectividade com backend de estado."""

    async def close(self) -> None:
        """Encerra conexoes do backend de estado."""
