"""Persistencia de sinais e resultados observados em Timescale/PostgreSQL."""

from __future__ import annotations

from typing import Any

import asyncpg

from indicator.domain import LiveObservation, Signal


class SignalStore:
    """Adapter SQL para historico auditavel do indicador."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.pool: Any = None

    async def start(self) -> None:
        """Abre pool e prepara tabelas idempotentes."""
        try:
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
                CREATE TABLE IF NOT EXISTS indicator_model_versions (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    PRIMARY KEY (symbol, period_seconds)
                );
                CREATE TABLE IF NOT EXISTS indicator_live (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
                    signal_epoch BIGINT NOT NULL, observed_at BIGINT NOT NULL,
                    spot DOUBLE PRECISION NOT NULL, delta DOUBLE PRECISION NOT NULL,
                    interim_side TEXT NOT NULL,
                    PRIMARY KEY (symbol, period_seconds)
                );
                """)
        except BaseException:
            await self.close()
            raise

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

    async def replace_models(self, versions: dict[tuple[str, int], str]) -> None:
        """Publica somente as versoes de modelo atualmente carregaveis."""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("DELETE FROM indicator_model_versions")
                for (symbol, period), version in versions.items():
                    await connection.execute(
                        "INSERT INTO indicator_model_versions (symbol, period_seconds, model_version) VALUES ($1,$2,$3)",
                        symbol,
                        period,
                        version,
                    )

    async def latest_signals(self) -> list[Signal]:
        """Restaura a última leitura emitida por par após reinício."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT DISTINCT ON (symbol, period_seconds)
                    symbol, period_seconds, candle_epoch, side, probability,
                    model_version, reason, close
                FROM indicator_signals
                ORDER BY symbol, period_seconds, candle_epoch DESC
                """
            )
        return [
            Signal(
                row["symbol"],
                row["period_seconds"],
                row["candle_epoch"],
                row["side"],
                row["probability"],
                row["model_version"],
                row["reason"],
                row["close"],
            )
            for row in rows
        ]

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

    async def save_live(self, observation: LiveObservation) -> None:
        """Atualiza somente o preço provisório mais recente de cada par."""
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO indicator_live
                (symbol, period_seconds, signal_epoch, observed_at, spot, delta, interim_side)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (symbol, period_seconds) DO UPDATE SET
                    signal_epoch=EXCLUDED.signal_epoch, observed_at=EXCLUDED.observed_at,
                    spot=EXCLUDED.spot, delta=EXCLUDED.delta,
                    interim_side=EXCLUDED.interim_side
                WHERE indicator_live.observed_at <= EXCLUDED.observed_at
                """,
                observation.symbol,
                observation.period,
                observation.signal_epoch,
                observation.observed_at,
                observation.spot,
                observation.delta,
                observation.interim_side,
            )

    async def close(self) -> None:
        """Fecha pool."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
