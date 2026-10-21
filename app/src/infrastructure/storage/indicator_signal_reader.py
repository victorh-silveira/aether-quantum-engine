"""Leitura isolada de sinais aprovados do banco do indicador."""

import asyncio

import asyncpg

from src.domain.models.indicator_consensus import ApprovedSignal, approved_signal_from_row


LATEST_M5_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (s.symbol) s.symbol, s.side, s.reason, s.probability,
           s.model_version, s.candle_epoch
    FROM indicator_signals s
    JOIN indicator_symbols a ON a.symbol = s.symbol AND a.active
    WHERE s.period_seconds = 300 AND s.symbol = ANY($1::text[])
    ORDER BY s.symbol, s.candle_epoch DESC
)
SELECT l.*, (m.model_version IS NOT NULL) AS model_active
FROM latest l
LEFT JOIN indicator_model_versions m
  ON m.symbol = l.symbol AND m.period_seconds = 300 AND m.model_version = l.model_version
"""

LATEST_PERIOD_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (s.symbol, s.period_seconds)
           s.symbol, s.period_seconds, s.side, s.reason, s.probability,
           s.model_version, s.candle_epoch
    FROM indicator_signals s
    JOIN indicator_symbols a ON a.symbol=s.symbol AND a.active
    WHERE s.period_seconds IN (300,900,3600)
    ORDER BY s.symbol, s.period_seconds, s.candle_epoch DESC
)
SELECT l.*, (m.model_version IS NOT NULL) AS model_active
FROM latest l
LEFT JOIN indicator_model_versions m
  ON m.symbol=l.symbol AND m.period_seconds=l.period_seconds AND m.model_version=l.model_version
"""

TRADE_SIGNALS_SQL = """
WITH selected AS (
    SELECT DISTINCT ON (r.symbol, r.period_seconds)
           r.symbol, r.period_seconds, r.setup_version
    FROM pullback_registry r
    JOIN indicator_symbols a ON a.symbol=r.symbol AND a.active
    WHERE (r.state=$1 OR ($1='piloto' AND r.state='ativo')) AND r.setup_version=$3 AND r.period_seconds IN (300, 900)
    ORDER BY r.symbol, r.period_seconds, r.created_at DESC, r.setup_version DESC
)
SELECT s.symbol, s.period_seconds, o.candle_epoch, o.setup_version AS model_version,
       o.side, NULL::double precision AS probability, o.reason, true AS model_active
FROM selected s
JOIN pullback_signals o ON o.symbol=s.symbol AND o.period_seconds=s.period_seconds
                   AND o.setup_version=s.setup_version
WHERE o.candle_epoch=($2::bigint / s.period_seconds) * s.period_seconds - s.period_seconds
"""


async def read_approved_m5_signals(
    dsn: str, symbols: list[str], *, expected_epoch: int, timeout: float = 3.0
) -> dict[str, ApprovedSignal]:
    """Consulta a ultima leitura de cada ativo; falha de banco propaga para o ciclo."""
    if not symbols:
        return {}
    connection = await asyncpg.connect(dsn, timeout=timeout)
    try:
        rows = await asyncio.wait_for(connection.fetch(LATEST_M5_SQL, symbols), timeout=timeout)
    finally:
        await connection.close()
    signals = (approved_signal_from_row(row, expected_epoch=expected_epoch) for row in rows)
    return {signal.symbol: signal for signal in signals if signal is not None}


async def read_approved_period_signals(
    dsn: str, *, now: int, timeout: float = 3.0
) -> dict[tuple[str, int], ApprovedSignal]:
    """Consulta sinais atuais de todos os periodos sem limitar o catalogo ao TCN."""
    connection = await asyncpg.connect(dsn, timeout=timeout)
    try:
        rows = await asyncio.wait_for(connection.fetch(LATEST_PERIOD_SQL), timeout=timeout)
    finally:
        await connection.close()
    signals = (
        approved_signal_from_row(
            row,
            expected_epoch=now // int(row["period_seconds"]) * int(row["period_seconds"]) - int(row["period_seconds"]),
        )
        for row in rows
    )
    return {(signal.symbol, signal.period): signal for signal in signals if signal is not None}


async def read_trade_signals(
    dsn: str,
    *,
    account_mode: str,
    now: int,
    setup_version: str | None = None,
    status_out: dict[tuple[str, int], str] | None = None,
    timeout: float = 3.0,
) -> dict[tuple[str, int], ApprovedSignal]:
    """Seleciona piloto DEMO ou versao ativa REAL por identidade de ativo e periodo."""
    if account_mode not in {"demo", "real"}:
        raise ValueError("modo de conta invalido")
    if not setup_version or not setup_version.startswith("pullback-1-"):
        return {}
    state = "piloto" if account_mode == "demo" else "ativo"
    connection = await asyncpg.connect(dsn, timeout=timeout)
    try:
        rows = await asyncio.wait_for(connection.fetch(TRADE_SIGNALS_SQL, state, now, setup_version), timeout=timeout)
    finally:
        await connection.close()
    if status_out is not None:
        status_out.update({(str(row["symbol"]), int(row["period_seconds"])): str(row["reason"]) for row in rows})
    signals = (
        approved_signal_from_row(
            row,
            expected_epoch=now // int(row["period_seconds"]) * int(row["period_seconds"]) - int(row["period_seconds"]),
        )
        for row in rows
    )
    return {(signal.symbol, signal.period): signal for signal in signals if signal is not None}
