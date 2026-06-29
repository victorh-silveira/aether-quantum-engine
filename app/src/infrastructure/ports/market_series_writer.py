"""Porta de escrita de series de mercado."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MarketSeriesWriter(Protocol):
    """Contrato async para ticks e barras OHLC."""

    async def enqueue_tick(self, *, symbol: str, epoch_ms: int, price: float) -> None:
        """Enfileira tick para persistencia assincrona."""

    async def enqueue_bar(self, *, symbol: str, bar: dict[str, Any]) -> None:
        """Enfileira barra OHLC com microestrutura."""

    async def flush(self) -> None:
        """Drena fila pendente."""

    async def ping(self) -> bool:
        """Valida conectividade."""

    async def close(self) -> None:
        """Encerra worker e pool."""
