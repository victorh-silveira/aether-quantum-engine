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
