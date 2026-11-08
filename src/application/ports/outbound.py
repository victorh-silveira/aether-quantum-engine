"""Ports de saida (hexagonal) — contratos Protocol sem acoplar infraestrutura."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, runtime_checkable

from src.domain.models.market_data import Candle


@runtime_checkable
class MarketCandlePort(Protocol):
    """Porta de leitura de velas OHLC para o dominio/aplicacao."""

    async def get_latest_candle(self, symbol: str, granularity: int) -> Candle | None:
        """Retorna a ultima vela fechada do simbolo/timeframe."""

    async def stream_candles(self, symbol: str, granularity: int) -> AsyncIterator[Candle]:
        """Stream assincrono de velas fechadas."""


@runtime_checkable
class SettlementQueuePort(Protocol):
    """Porta da fila de liquidacao (Redis ZSET no adapter)."""

    async def enqueue(self, contract_id: str, score: float, payload: Mapping[str, Any]) -> bool:
        """Enfileira contrato de forma idempotente; True se novo."""

    async def pop_due(self, score: float, limit: int = 32) -> list[Mapping[str, Any]]:
        """Remove e retorna contratos vencidos ate score temporal."""


@runtime_checkable
class ModelArtifactPort(Protocol):
    """Porta de artefatos de modelo (MinIO/local no adapter)."""

    async def load_bytes(self, key: str) -> bytes | None:
        """Carrega bytes do artefato ou None se ausente."""

    async def put_bytes(self, key: str, data: bytes) -> None:
        """Persiste bytes do artefato."""
