"""Writer nulo para testes e modo sem Timescale."""

from __future__ import annotations

from typing import Any


class NullMarketWriter:
    """Descarta ticks e barras sem I/O."""

    async def enqueue_tick(self, *, symbol: str, epoch_ms: int, price: float) -> None:
        """Ignora tick recebido."""
        _ = (symbol, epoch_ms, price)

    async def enqueue_bar(self, *, symbol: str, bar: dict[str, Any]) -> None:
        """Ignora barra recebida."""
        _ = (symbol, bar)

    async def flush(self) -> None:
        """Nao possui buffer pendente."""
        return

    async def ping(self) -> bool:
        """Sempre disponivel em modo nulo."""
        return True

    async def close(self) -> None:
        """Encerramento sem efeito."""
        return
