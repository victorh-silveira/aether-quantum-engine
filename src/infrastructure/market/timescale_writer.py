"""Persistencia assincrona de ticks e barras em TimescaleDB."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import asyncpg

from src.infrastructure.market.timescale_contracts_sql import (
    CONTRACT_FIELDS as _CONTRACT_FIELDS,
    CONTRACT_UPSERT_SQL as _CONTRACT_SQL,
)


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
        self._last_error: Exception | None = None
        self.logger = logging.getLogger("AETH")

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Cria pool asyncpg sob demanda."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=4,
                statement_cache_size=64,
                command_timeout=30.0,
            )
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

    async def enqueue_contract_audit(self, row: dict[str, Any]) -> None:
        """Enfileira dados confirmados de compra/liquidacao sem inventar spots."""
        if self._closed:
            return
        self._queue.put_nowait(
            (
                "contract",
                tuple(
                    (row.get(key) or "pending") if key == "settlement_source" else row.get(key)
                    for key in _CONTRACT_FIELDS
                ),
            )
        )
        self._ensure_worker()

    async def record_quote_observation(self, row: dict[str, Any]) -> None:
        """Persiste cotacao observada antes de uma eventual compra."""
        if self._closed:
            raise RuntimeError("captura Timescale encerrada")
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO indicator_quote_observations
                    (observation_id, account_mode, symbol, period_seconds, candle_epoch,
                     model_version, side, proposal_id, side_probability, quote_price,
                     quote_payout, quote_ev, cohort)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                uuid4(),
                row["account_mode"],
                row["symbol"],
                row["period_seconds"],
                row["candle_epoch"],
                row["model_version"],
                row["side"],
                row["proposal_id"],
                row["side_probability"],
                row["quote_price"],
                row["quote_payout"],
                row["quote_ev"],
                row["cohort"],
            )

    async def _run_worker(self) -> None:
        """Consome fila e faz flush em lotes por intervalo ou tamanho."""
        tick_batch: list[tuple[Any, ...]] = []
        bar_batch: list[tuple[Any, ...]] = []
        contract_batch: list[tuple[Any, ...]] = []
        try:
            while not self._closed or not self._queue.empty():
                deadline = asyncio.get_running_loop().time() + self._flush_interval
                while len(tick_batch) + len(bar_batch) + len(contract_batch) < self._batch_limit:
                    timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                    if timeout <= 0:
                        break
                    try:
                        kind, row = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    except TimeoutError:
                        break
                    if kind == "tick":
                        tick_batch.append(row)
                    elif kind == "contract":
                        contract_batch.append(row)
                    else:
                        bar_batch.append(row)
                if tick_batch or bar_batch or contract_batch:
                    await self._flush_batches(tick_batch, bar_batch, contract_batch)
                    tick_batch.clear()
                    bar_batch.clear()
                    contract_batch.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = exc
            self.logger.error("TSDB: worker falhou: %s", exc)

    async def _flush_batches(
        self,
        tick_batch: list[tuple[Any, ...]],
        bar_batch: list[tuple[Any, ...]],
        contract_batch: list[tuple[Any, ...]] | None = None,
    ) -> None:
        """Executa batch insert de ticks e barras no TimescaleDB."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if tick_batch:
                await conn.executemany(_TICK_SQL, tick_batch)
            if bar_batch:
                await conn.executemany(_BAR_SQL, bar_batch)
            if contract_batch:
                await conn.executemany(_CONTRACT_SQL, contract_batch)
        self._last_error = None

    async def flush(self) -> None:
        """Esvazia fila pendente e grava imediatamente."""
        if self._worker is not None and not self._worker.done():
            await asyncio.sleep(self._flush_interval * 1.5)
        tick_batch: list[tuple[Any, ...]] = []
        bar_batch: list[tuple[Any, ...]] = []
        contract_batch: list[tuple[Any, ...]] = []
        while not self._queue.empty():
            kind, row = self._queue.get_nowait()
            if kind == "tick":
                tick_batch.append(row)
            elif kind == "contract":
                contract_batch.append(row)
            else:
                bar_batch.append(row)
        if tick_batch or bar_batch or contract_batch:
            await self._flush_batches(tick_batch, bar_batch, contract_batch)

    async def fetch_inferred_rest_contract_ids(self, *, limit: int = 50) -> list[int]:
        """Lista contract_id com liquidacao inferred_rest aguardando confirmacao oficial."""
        if self._closed:
            return []
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT contract_id FROM contract_executions WHERE settlement_source='inferred_rest' "
                    "ORDER BY updated_at DESC LIMIT $1",
                    int(limit),
                )
            return [int(r["contract_id"]) for r in rows]
        except Exception:
            return []

    async def ping(self) -> bool:
        """Valida conectividade com SELECT 1."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return int(val) == 1

    async def close(self) -> None:
        """Aguarda ultimo batch antes de fechar o pool, sem cancelar a gravacao."""
        self._closed = True
        if self._worker is not None:
            await self._worker
        await self.flush()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
