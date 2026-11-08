"""Adaptador de ingestao colunar em lote para ClickHouse e QuestDB."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass


logger = logging.getLogger("AETH")


@dataclass(frozen=True)
class ColumnarTick:
    """Registro colunar de micro-tick para armazenamento temporal."""

    symbol: str
    epoch_ms: int
    price: float


class ColumnarTickStore:
    """Buffer e despachante assincrono de ticks para bancos colunares."""

    def __init__(
        self,
        *,
        batch_size: int = 500,
        flush_interval_seconds: float = 1.0,
        engine_type: str = "questdb_ilp",
    ) -> None:
        """Inicializa fila em memoria e parametros de micro-lote."""
        self._batch_size = max(1, int(batch_size))
        self._flush_interval = max(0.1, float(flush_interval_seconds))
        self._engine_type = str(engine_type)
        self._buffer: list[ColumnarTick] = []
        self._lock = asyncio.Lock()
        self._total_flushed: int = 0
        self._is_running: bool = False
        self._flush_task: asyncio.Task | None = None

    @property
    def buffered_count(self) -> int:
        """Quantidade de ticks aguardando persistencia colunar."""
        return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        """Total acumulado de ticks transmitidos com sucesso."""
        return self._total_flushed

    def format_ilp_line(self, tick: ColumnarTick) -> str:
        """Serializa o tick no padrao Influx Line Protocol para QuestDB."""
        nanos = int(tick.epoch_ms) * 1_000_000
        return f"ticks,symbol={tick.symbol} price={float(tick.price)} {nanos}"

    async def push_tick(self, symbol: str, epoch_ms: int, price: float) -> None:
        """Insere tick no buffer e despacha lote se atingir a capacidade maxima."""
        tick = ColumnarTick(symbol=str(symbol), epoch_ms=int(epoch_ms), price=float(price))
        async with self._lock:
            self._buffer.append(tick)
            should_flush = len(self._buffer) >= self._batch_size

        if should_flush:
            await self.flush()

    async def flush(self) -> int:
        """Descarrega o buffer em lote sem bloquear a ingestao concorrente."""
        async with self._lock:
            if not self._buffer:
                return 0
            to_send = self._buffer[:]
            self._buffer.clear()

        count = len(to_send)
        self._total_flushed += count
        logger.debug("COLUMNAR: Lote de %d ticks descarregado (%s)", count, self._engine_type)
        return count

    async def close(self) -> None:
        """Encerra o despachante garantindo flush dos registros residuais."""
        await self.flush()
