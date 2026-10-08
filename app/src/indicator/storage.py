"""Persistencia de sinais e resultados observados em Timescale/PostgreSQL."""

from __future__ import annotations

from typing import Any

import asyncpg

from indicator.domain import Signal


class SignalStore:
    """Adapter SQL para historico auditavel do indicador."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.pool: Any = None

    async def start(self) -> None:
        """Abre pool e prepara tabelas idempotentes."""
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)
        async with self.pool.acquire() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS indicator_symbols (
                    symbol TEXT PRIMARY KEY, name TEXT NOT NULL, active BOOLEAN NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS indicator_signals (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL, candle_epoch BIGINT NOT NULL,
                    side TEXT NOT NULL, probability DOUBLE PRECISION, model_version TEXT,
                    reason TEXT NOT NULL, close DOUBLE PRECISION NOT NULL,
                    observed_side TEXT, correct BOOLEAN,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (symbol, period_seconds, candle_epoch)
                );
            """)

    async def set_symbols(self, symbols: dict[str, str]) -> None:
        """Atualiza catalogo e desativa instrumentos ausentes."""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("UPDATE indicator_symbols SET active = false, updated_at = now()")
                for symbol, name in symbols.items():
                    await connection.execute(
                        """
                        INSERT INTO indicator_symbols (symbol, name, active) VALUES ($1, $2, true)
                        ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name, active = true, updated_at = now()
                    """,
                        symbol,
                        name,
                    )

    async def save(self, signal: Signal, observed_side: str | None = None) -> None:
        """Grava leitura e confirma resultado da leitura anterior, quando conhecido."""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                if observed_side is not None:
                    await connection.execute(
                        """
                        UPDATE indicator_signals SET observed_side = $1,
                            correct = CASE WHEN $1 IN ('CALL', 'PUT') AND side IN ('CALL', 'PUT')
                                THEN side = $1 ELSE NULL END
                        WHERE symbol = $2 AND period_seconds = $3 AND candle_epoch = $4
                    """,
                        observed_side,
                        signal.symbol,
                        signal.period,
                        signal.candle_epoch - signal.period,
                    )
                await connection.execute(
                    """
                    INSERT INTO indicator_signals
                    (symbol, period_seconds, candle_epoch, side, probability, model_version, reason, close)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT (symbol, period_seconds, candle_epoch) DO NOTHING
                """,
                    signal.symbol,
                    signal.period,
                    signal.candle_epoch,
                    signal.side,
                    signal.probability,
                    signal.model_version,
                    signal.reason,
                    signal.close,
                )

    async def close(self) -> None:
        """Fecha pool."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
