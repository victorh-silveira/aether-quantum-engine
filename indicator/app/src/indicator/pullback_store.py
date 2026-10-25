"""Persistencia isolada do setup e de sua promocao por contratos DEMO."""

from __future__ import annotations

from indicator.domain import PERIODS, Signal
from indicator.pullback import VERSION
from indicator.storage import SignalStore


class PullbackStore(SignalStore):
    """Nao mistura sinais ou aptidao do setup com modelos estatisticos antigos."""

    async def start(self) -> None:
        """Prepara o esquema versionado apos a conexao com Timescale."""
        await super().start()
        async with self.pool.acquire() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS pullback_signals (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL, candle_epoch BIGINT NOT NULL,
                    side TEXT NOT NULL, reason TEXT NOT NULL, setup_version TEXT NOT NULL,
                    close DOUBLE PRECISION NOT NULL, recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (symbol, period_seconds, candle_epoch, setup_version)
                );
                CREATE TABLE IF NOT EXISTS pullback_registry (
                    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL, setup_version TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'piloto', created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (symbol, period_seconds, setup_version),
                    CHECK (state IN ('piloto','ativo','reprovado'))
                );
                CREATE OR REPLACE VIEW pullback_broker_status AS
                WITH ranked AS (
                    SELECT symbol, indicator_period_seconds AS period_seconds,
                        indicator_model_version AS setup_version, contract_id, buy_price, profit,
                        settlement_source, row_number() OVER (
                            PARTITION BY symbol, indicator_period_seconds, indicator_model_version
                            ORDER BY date_start NULLS LAST, contract_id
                        ) AS ordinal
                    FROM contract_executions
                    WHERE account_mode='demo' AND quote_cohort='setup_pilot'
                        AND indicator_model_version LIKE 'pullback-1-%'
                        AND buy_price > 0 AND coalesce(contract_type, direction) IN ('CALL','PUT')
                ), samples AS (
                    SELECT symbol, period_seconds, setup_version,
                        count(*) AS purchased,
                        count(*) FILTER (WHERE settlement_source='broker' AND profit IS NOT NULL) AS settled,
                        avg(profit / buy_price) FILTER (
                            WHERE settlement_source='broker' AND profit IS NOT NULL) AS mean_return,
                        stddev_samp(profit / buy_price) FILTER (
                            WHERE settlement_source='broker' AND profit IS NOT NULL) AS return_sd,
                        count(*) FILTER (
                            WHERE settlement_source='broker' AND profit > 0) AS wins
                    FROM ranked WHERE ordinal <= 200
                    GROUP BY symbol, period_seconds, setup_version
                )
                SELECT *, CASE WHEN settled >= 2 THEN
                    mean_return - 1.653 * return_sd / sqrt(settled)
                END AS lower_95 FROM samples;
            """)

    async def register_setups(self, symbols: dict[str, str], version: str = VERSION) -> None:
        """Registra versoes atuais sem sobrescrever decisoes anteriores."""
        async with self.pool.acquire() as connection:
            for symbol in symbols:
                for period in PERIODS:
                    await connection.execute(
                        """INSERT INTO pullback_registry(symbol,period_seconds,setup_version)
                        VALUES ($1,$2,$3) ON CONFLICT DO NOTHING""",
                        symbol,
                        period,
                        version,
                    )

    async def promote_setups(self) -> None:
        """Promove ou reprova apos as primeiras 200 liquidacoes unicas do broker."""
        async with self.pool.acquire() as connection:
            await connection.execute("""
                UPDATE pullback_registry r
                SET state=CASE WHEN e.lower_95 > 0 THEN 'ativo' ELSE 'reprovado' END,
                    updated_at=now()
                FROM pullback_broker_status e
                WHERE r.symbol=e.symbol AND r.period_seconds=e.period_seconds
                    AND r.setup_version=e.setup_version AND r.state='piloto'
                    AND e.purchased >= 200 AND e.settled >= 200
            """)

    async def promote_symbol_setup(self, symbol: str, period: int = 900, version: str = VERSION) -> None:
        """Promove o setup do simbolo especifico para o estado ativo no registro."""
        async with self.pool.acquire() as connection:
            await connection.execute(
                """INSERT INTO pullback_registry(symbol, period_seconds, setup_version, state, updated_at)
                VALUES ($1, $2, $3, 'ativo', now())
                ON CONFLICT (symbol, period_seconds, setup_version)
                DO UPDATE SET state = 'ativo', updated_at = now()""",
                symbol,
                period,
                version,
            )

    async def save_setup(self, signal: Signal) -> None:
        """Grava setup e motivo sem inventar probabilidade direcional."""
        async with self.pool.acquire() as connection:
            await connection.execute(
                """INSERT INTO pullback_signals
                    (symbol,period_seconds,candle_epoch,side,reason,setup_version,close)
                    VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING""",
                signal.symbol,
                signal.period,
                signal.candle_epoch,
                signal.side,
                signal.reason,
                signal.model_version,
                signal.close,
            )

    async def latest_setups(self) -> list[Signal]:
        """Restaura leituras apenas da versao fixa vigente."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT DISTINCT ON (symbol, period_seconds) symbol, period_seconds,
                    candle_epoch, side, reason, setup_version, close
                FROM pullback_signals WHERE setup_version=$1 OR setup_version LIKE 'pullback-1-%'
                ORDER BY symbol, period_seconds, candle_epoch DESC
            """,
                VERSION,
            )
        return [
            Signal(
                row["symbol"],
                row["period_seconds"],
                row["candle_epoch"],
                row["side"],
                None,
                row["setup_version"],
                row["reason"],
                row["close"],
            )
            for row in rows
        ]
