"""Leitura de compras e liquidacoes DEMO atribuidas ao indicador."""

from __future__ import annotations

import asyncio

import asyncpg

from src.domain.risk.broker_ev_evidence import BrokerEvEvidence


EVIDENCE_SQL = """
SELECT symbol, indicator_period_seconds, indicator_model_version, contract_id,
       buy_price, profit, settlement_source, quote_price, quote_payout, quote_ev, quote_cohort
FROM contract_executions
WHERE account_mode = 'demo' AND symbol = ANY($1::text[])
  AND indicator_model_version IS NOT NULL
  AND indicator_period_seconds IN (300,900,3600)
  AND buy_price > 0 AND coalesce(contract_type, direction) IN ('CALL', 'PUT')
ORDER BY symbol, indicator_period_seconds, indicator_model_version, date_start NULLS LAST, contract_id
"""

REAL_EVIDENCE_SQL = """
WITH recent AS (
    SELECT symbol, indicator_period_seconds, indicator_model_version,
           contract_id, profit, buy_price,
           row_number() OVER (
               PARTITION BY symbol, indicator_period_seconds, indicator_model_version
               ORDER BY date_start DESC NULLS LAST, contract_id DESC
           ) AS rank
    FROM contract_executions
    WHERE account_mode='real' AND symbol = ANY($1::text[])
      AND indicator_period_seconds IN (300,900,3600)
      AND indicator_model_version IS NOT NULL
      AND settlement_source='broker' AND profit IS NOT NULL AND buy_price > 0
)
SELECT symbol, indicator_period_seconds, indicator_model_version, contract_id, profit, buy_price
FROM recent WHERE rank <= 200
ORDER BY symbol, indicator_period_seconds, indicator_model_version, rank
"""


async def read_indicator_broker_evidence(
    dsn: str, symbols: list[str], *, timeout: float = 3.0
) -> dict[tuple[str, int, str], BrokerEvEvidence]:
    """Agrega IDs unicos; somente profit confirmado pelo broker qualifica."""
    if not symbols:
        return {}
    connection = await asyncpg.connect(dsn, timeout=timeout)
    try:
        rows = await asyncio.wait_for(connection.fetch(EVIDENCE_SQL, symbols), timeout=timeout)
    finally:
        await connection.close()
    grouped: dict[tuple[str, int, str], tuple[set[int], list[float], list[int]]] = {}
    for row in rows:
        key = (str(row["symbol"]), int(row["indicator_period_seconds"]), str(row["indicator_model_version"]))
        seen, returns, counts = grouped.setdefault(key, (set(), [], [0, 0]))
        contract_id = int(row["contract_id"])
        if contract_id in seen:
            continue
        seen.add(contract_id)
        if key[2].startswith("pullback-1-"):
            if row["quote_cohort"] != "setup_pilot":
                continue
            if counts[0] >= 200:
                continue
            counts[0] += 1
            if row["settlement_source"] == "broker" and row["profit"] is not None:
                returns.append(float(row["profit"]) / float(row["buy_price"]))
            continue
        try:
            price = float(row["quote_price"])
            payout = float(row["quote_payout"])
            edge = float(row["quote_ev"])
            confirmatory = (
                row["quote_cohort"] == "confirmatory"
                and price > 0
                and payout > price
                and abs(price - float(row["buy_price"])) <= 0.01
                and edge > 0
                and edge >= 0.01 * payout / price - 1e-9
            )
        except (TypeError, ValueError):
            confirmatory = False
        if not confirmatory:
            counts[1] += 1
            continue
        counts[0] += 1
        if row["settlement_source"] == "broker" and row["profit"] is not None:
            returns.append(float(row["profit"]) / float(row["buy_price"]))
    return {
        key: BrokerEvEvidence(counts[0], tuple(returns), exploratory_purchased=counts[1])
        for key, (_, returns, counts) in grouped.items()
    }


async def read_indicator_real_evidence(
    dsn: str, symbols: list[str], *, timeout: float = 3.0
) -> dict[tuple[str, int, str], BrokerEvEvidence]:
    """Le ate 200 liquidacoes REAL recentes, atribuidas e confirmadas."""
    if not symbols:
        return {}
    connection = await asyncpg.connect(dsn, timeout=timeout)
    try:
        rows = await asyncio.wait_for(connection.fetch(REAL_EVIDENCE_SQL, symbols), timeout=timeout)
    finally:
        await connection.close()
    grouped: dict[tuple[str, int, str], dict[int, float]] = {}
    for row in rows:
        key = (str(row["symbol"]), int(row["indicator_period_seconds"]), str(row["indicator_model_version"]))
        grouped.setdefault(key, {})[int(row["contract_id"])] = float(row["profit"]) / float(row["buy_price"])
    return {key: BrokerEvEvidence(len(values), tuple(values.values())) for key, values in grouped.items()}
