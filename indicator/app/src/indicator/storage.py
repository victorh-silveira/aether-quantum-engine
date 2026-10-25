"""Persistencia de sinais e resultados observados em Timescale/PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import asyncpg

from indicator.domain import Candle, LiveObservation, Signal


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
                CREATE TABLE IF NOT EXISTS indicator_candles (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
                    epoch BIGINT NOT NULL, time TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (symbol, period_seconds, epoch)
                );
                CREATE TABLE IF NOT EXISTS indicator_model_registry (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
                    model_version TEXT NOT NULL, state TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (symbol, period_seconds, model_version),
                    CHECK (state IN ('pesquisa', 'piloto', 'ativo', 'reprovado'))
                );
                CREATE TABLE IF NOT EXISTS indicator_candidate_signals (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
                    candle_epoch BIGINT NOT NULL, model_version TEXT NOT NULL,
                    side TEXT NOT NULL, probability DOUBLE PRECISION,
                    reason TEXT NOT NULL, close DOUBLE PRECISION NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (symbol, period_seconds, candle_epoch, model_version)
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

    async def save_candles(self, symbol: str, period: int, candles: list[Candle]) -> None:
        """Persiste velas diretas com instante canonico e chave unica."""
        if period not in (300, 900, 3600, 86400):
            raise ValueError("periodo nao suportado")
        rows = [
            (
                symbol,
                period,
                candle.epoch,
                datetime.fromtimestamp(candle.epoch, UTC),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
            )
            for candle in candles
        ]
        async with self.pool.acquire() as connection:
            await connection.executemany(
                """
                INSERT INTO indicator_candles
                    (symbol, period_seconds, epoch, time, open, high, low, close)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (symbol, period_seconds, epoch) DO UPDATE SET
                    time=EXCLUDED.time, open=EXCLUDED.open, high=EXCLUDED.high,
                    low=EXCLUDED.low, close=EXCLUDED.close
                """,
                rows,
            )

    async def register_model(self, symbol: str, period: int, version: str, state: str) -> None:
        """Registra identidade imutavel e estado inicial do desafiante."""
        if state not in {"pesquisa", "piloto", "reprovado"}:
            raise ValueError("estado inicial invalido")
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO indicator_model_registry(symbol, period_seconds, model_version, state)
                VALUES ($1,$2,$3,$4)
                ON CONFLICT (symbol, period_seconds, model_version) DO NOTHING
                """,
                symbol,
                period,
                version,
                state,
            )

    async def promote_models(self) -> None:
        """Conclui pilotos por evidencia confirmatoria sem substituir outro par."""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
                    FROM indicator_broker_ev_status e
                    WHERE r.symbol=e.symbol AND r.period_seconds=e.indicator_period_seconds
                      AND r.model_version=e.indicator_model_version AND r.state='piloto'
                      AND e.purchased >= 200 AND e.settled >= 200 AND NOT e.promoted
                    """
                )
                await connection.execute(
                    """
                    UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
                    WHERE r.state='ativo' AND EXISTS (
                        SELECT 1 FROM indicator_model_registry pilot
                        JOIN indicator_broker_ev_status e ON e.symbol=pilot.symbol
                          AND e.indicator_period_seconds=pilot.period_seconds
                          AND e.indicator_model_version=pilot.model_version
                        WHERE pilot.symbol=r.symbol AND pilot.period_seconds=r.period_seconds
                          AND pilot.state='piloto' AND e.promoted
                    )
                    """
                )
                await connection.execute(
                    """
                    UPDATE indicator_model_registry r SET state='ativo', updated_at=now()
                    FROM (
                        SELECT DISTINCT ON (p.symbol, p.period_seconds)
                            p.symbol, p.period_seconds, p.model_version
                        FROM indicator_model_registry p
                        JOIN indicator_broker_ev_status e ON e.symbol=p.symbol
                          AND e.indicator_period_seconds=p.period_seconds
                          AND e.indicator_model_version=p.model_version
                        WHERE p.state='piloto' AND e.promoted
                        ORDER BY p.symbol, p.period_seconds, p.created_at ASC, p.model_version
                    ) selected
                    WHERE r.symbol=selected.symbol AND r.period_seconds=selected.period_seconds
                      AND r.model_version=selected.model_version AND r.state='piloto'
                    """
                )
                await connection.execute(
                    """
                    UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
                    FROM indicator_broker_ev_status e
                    WHERE r.symbol=e.symbol AND r.period_seconds=e.indicator_period_seconds
                      AND r.model_version=e.indicator_model_version AND r.state='piloto'
                      AND e.promoted AND EXISTS (
                          SELECT 1 FROM indicator_model_registry active
                          WHERE active.symbol=r.symbol AND active.period_seconds=r.period_seconds
                            AND active.state='ativo'
                      )
                    """
                )

    async def trade_models(self) -> dict[tuple[str, int, str], str]:
        """Seleciona uma unica versao por ativo, periodo e estado negociavel."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT DISTINCT ON (symbol, period_seconds, state)
                    symbol, period_seconds, state, model_version
                FROM indicator_model_registry WHERE state IN ('piloto','ativo')
                ORDER BY symbol, period_seconds, state, created_at ASC, model_version
                """
            )
        return {
            (str(row["symbol"]), int(row["period_seconds"]), str(row["state"])): str(row["model_version"])
            for row in rows
        }

    async def save_candidate(self, signal: Signal) -> None:
        """Persiste um sinal versionado sem alterar o sinal principal do painel."""
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO indicator_candidate_signals
                    (symbol, period_seconds, candle_epoch, model_version, side, probability, reason, close)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (symbol, period_seconds, candle_epoch, model_version) DO NOTHING
                """,
                signal.symbol,
                signal.period,
                signal.candle_epoch,
                signal.model_version,
                signal.side,
                signal.probability,
                signal.reason,
                signal.close,
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

    async def retained_broker_versions(self) -> dict[tuple[str, int], str]:
        """Mantem a versao principal enquanto ela estiver em piloto ou ativa."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT m.symbol, m.period_seconds, m.model_version
                FROM indicator_model_versions m
                JOIN indicator_model_registry r ON r.symbol=m.symbol
                  AND r.period_seconds=m.period_seconds AND r.model_version=m.model_version
                LEFT JOIN indicator_broker_ev_status e ON e.symbol=r.symbol
                  AND e.indicator_period_seconds=r.period_seconds
                  AND e.indicator_model_version=r.model_version
                WHERE r.state='ativo' OR (r.state='piloto'
                    AND (coalesce(e.purchased,0) < 200 OR e.settled < e.purchased))
                """
            )
        return {(str(row["symbol"]), int(row["period_seconds"])): str(row["model_version"]) for row in rows}

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
