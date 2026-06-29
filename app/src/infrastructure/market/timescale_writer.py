"""Persistencia assincrona de ticks e barras em TimescaleDB."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

import asyncpg


_TICK_SQL = "INSERT INTO ticks (time, symbol, epoch_ms, price) VALUES ($1, $2, $3, $4)"
_BAR_SQL = (
    "INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close, "
    "tick_count, mean_inter_tick_ms, price_velocity) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
    "ON CONFLICT DO NOTHING"
)


class TimescaleMarketWriter:
    """Fila asyncio com batch insert para TimescaleDB."""

    def __init__(
        self,
        *,
        dsn: str,
        flush_interval_ms: float = 200.0,
        batch_limit: int = 500,
    ):
        self._dsn = dsn
        self._flush_interval = max(0.05, float(flush_interval_ms) / 1000.0)
        self._batch_limit = max(1, int(batch_limit))
        self._pool: asyncpg.Pool | None = None
        self._queue: asyncio.Queue[tuple[str, tuple[Any, ...]]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._closed = False
        self.logger = logging.getLogger("AETH")

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Cria pool asyncpg sob demanda."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        return self._pool

    def _ensure_worker(self) -> None:
        """Inicia task de batch insert se necessario."""
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def enqueue_tick(self, *, symbol: str, epoch_ms: int, price: float) -> None:
        """Enfileira tick para persistencia assincrona."""
        if self._closed:
            return
        ts = datetime.fromtimestamp(int(epoch_ms) / 1000.0, tz=UTC)
        self._queue.put_nowait(("tick", (ts, str(symbol), int(epoch_ms), float(price))))
        self._ensure_worker()

    async def enqueue_bar(self, *, symbol: str, bar: dict[str, Any]) -> None:
        """Enfileira barra OHLC para persistencia assincrona."""
        if self._closed:
            return
        epoch = int(bar.get("epoch", 0))
        ts = datetime.fromtimestamp(epoch, tz=UTC)
        row = (
            ts,
            str(symbol),
            epoch,
            int(bar.get("granularity", 0)),
            bar.get("open"),
            bar.get("high"),
            bar.get("low"),
            bar.get("close"),
            bar.get("tick_count"),
            bar.get("mean_inter_tick_ms"),
            bar.get("price_velocity"),
        )
        self._queue.put_nowait(("bar", row))
        self._ensure_worker()

    async def _run_worker(self) -> None:
        """Consome fila e faz flush em lotes por intervalo ou tamanho."""
        tick_batch: list[tuple[Any, ...]] = []
        bar_batch: list[tuple[Any, ...]] = []
        try:
            while not self._closed or not self._queue.empty():
                deadline = asyncio.get_running_loop().time() + self._flush_interval
                while len(tick_batch) + len(bar_batch) < self._batch_limit:
                    timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                    if timeout <= 0:
                        break
                    try:
                        kind, row = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    except TimeoutError:
                        break
                    if kind == "tick":
                        tick_batch.append(row)
                    else:
                        bar_batch.append(row)
                if tick_batch or bar_batch:
                    await self._flush_batches(tick_batch, bar_batch)
                    tick_batch.clear()
                    bar_batch.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error("TSDB: worker falhou: %s", exc)

    async def _flush_batches(
        self,
        tick_batch: list[tuple[Any, ...]],
        bar_batch: list[tuple[Any, ...]],
    ) -> None:
        """Executa batch insert de ticks e barras no TimescaleDB."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if tick_batch:
                await conn.executemany(_TICK_SQL, tick_batch)
            if bar_batch:
                await conn.executemany(_BAR_SQL, bar_batch)

    async def flush(self) -> None:
        """Esvazia fila pendente e grava imediatamente."""
        if self._worker is not None and not self._worker.done():
            await asyncio.sleep(self._flush_interval * 1.5)
        tick_batch: list[tuple[Any, ...]] = []
        bar_batch: list[tuple[Any, ...]] = []
        while not self._queue.empty():
            kind, row = self._queue.get_nowait()
            if kind == "tick":
                tick_batch.append(row)
            else:
                bar_batch.append(row)
        if tick_batch or bar_batch:
            await self._flush_batches(tick_batch, bar_batch)

    async def ping(self) -> bool:
        """Valida conectividade com SELECT 1."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return int(val) == 1

    async def close(self) -> None:
        """Cancela worker, faz flush final e fecha pool."""
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        await self.flush()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
