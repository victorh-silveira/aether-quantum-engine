"""Leitura da amostra DEMO por versao sem duplicar contratos."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.storage import indicator_broker_evidence_reader as reader


@pytest.mark.asyncio
async def test_reader_counts_buys_once_and_only_broker_settlements(monkeypatch):
    rows = [
        {
            "symbol": "R_25",
            "indicator_period_seconds": 300,
            "indicator_model_version": "v1",
            "contract_id": 1,
            "buy_price": 10,
            "profit": 8,
            "settlement_source": "broker",
            "quote_price": 10,
            "quote_payout": 18,
            "quote_ev": 0.2,
            "quote_cohort": "confirmatory",
        },
        {
            "symbol": "R_25",
            "indicator_period_seconds": 300,
            "indicator_model_version": "v1",
            "contract_id": 1,
            "buy_price": 10,
            "profit": 8,
            "settlement_source": "broker",
            "quote_price": 10,
            "quote_payout": 18,
            "quote_ev": 0.2,
            "quote_cohort": "confirmatory",
        },
        {
            "symbol": "R_25",
            "indicator_period_seconds": 300,
            "indicator_model_version": "v1",
            "contract_id": 2,
            "buy_price": 10,
            "profit": None,
            "settlement_source": "pending",
            "quote_price": None,
            "quote_payout": None,
            "quote_ev": None,
            "quote_cohort": "exploratory",
        },
        {
            "symbol": "R_25",
            "indicator_period_seconds": 900,
            "indicator_model_version": "v2",
            "contract_id": 3,
            "buy_price": 10,
            "profit": 8,
            "settlement_source": "profit_table",
            "quote_price": 10,
            "quote_payout": 18,
            "quote_ev": 0.2,
            "quote_cohort": "confirmatory",
        },
    ]
    connection = SimpleNamespace(fetch=AsyncMock(return_value=rows), close=AsyncMock())
    monkeypatch.setattr(reader.asyncpg, "connect", AsyncMock(return_value=connection))
    result = await reader.read_indicator_broker_evidence("dsn", ["R_25"])
    assert result[("R_25", 300, "v1")].purchased == 1
    assert result[("R_25", 300, "v1")].exploratory_purchased == 1
    assert result[("R_25", 300, "v1")].returns == (0.8,)
    assert result[("R_25", 900, "v2")].returns == ()
    connection.close.assert_awaited_once()
    assert await reader.read_indicator_broker_evidence("dsn", []) == {}


@pytest.mark.asyncio
async def test_real_reader_limits_and_deduplicates_by_version(monkeypatch):
    rows = [
        {
            "symbol": "R_25",
            "indicator_period_seconds": 900,
            "indicator_model_version": "v3",
            "contract_id": contract_id,
            "profit": -2,
            "buy_price": 10,
        }
        for contract_id in (7, 7, 8)
    ]
    connection = SimpleNamespace(fetch=AsyncMock(return_value=rows), close=AsyncMock())
    monkeypatch.setattr(reader.asyncpg, "connect", AsyncMock(return_value=connection))
    evidence = await reader.read_indicator_real_evidence("dsn", ["R_25"])
    assert evidence[("R_25", 900, "v3")].returns == (-0.2, -0.2)
    assert "rank <= 200" in connection.fetch.await_args.args[0]
    assert await reader.read_indicator_real_evidence("dsn", []) == {}


@pytest.mark.asyncio
async def test_setup_pilot_counts_first_unique_broker_contracts_without_ev(monkeypatch):
    version = "pullback-1-fixed"
    rows = [
        {
            "symbol": "R_25",
            "indicator_period_seconds": 300,
            "indicator_model_version": version,
            "contract_id": contract_id,
            "buy_price": 1,
            "profit": profit,
            "settlement_source": source,
            "quote_price": 1,
            "quote_payout": 1.8,
            "quote_ev": -0.2,
            "quote_cohort": cohort,
        }
        for contract_id, profit, source, cohort in (
            (1, 0.8, "broker", "setup_pilot"),
            (1, 0.8, "broker", "setup_pilot"),
            (2, None, "pending", "setup_pilot"),
            (3, 0.8, "broker", "old"),
        )
    ]
    connection = SimpleNamespace(fetch=AsyncMock(return_value=rows), close=AsyncMock())
    monkeypatch.setattr(reader.asyncpg, "connect", AsyncMock(return_value=connection))
    result = await reader.read_indicator_broker_evidence("dsn", ["R_25"])
    sample = result[("R_25", 300, version)]
    assert sample.purchased == 2
    assert sample.returns == (0.8,)
    assert sample.pilot_available


@pytest.mark.asyncio
async def test_setup_pilot_caps_first_200_purchases(monkeypatch):
    rows = [
        {
            "symbol": "R_25",
            "indicator_period_seconds": 300,
            "indicator_model_version": "pullback-1-fixed",
            "contract_id": contract_id,
            "buy_price": 1,
            "profit": 0.8,
            "settlement_source": "broker",
            "quote_cohort": "setup_pilot",
        }
        for contract_id in range(1, 202)
    ]
    connection = SimpleNamespace(fetch=AsyncMock(return_value=rows), close=AsyncMock())
    monkeypatch.setattr(reader.asyncpg, "connect", AsyncMock(return_value=connection))
    sample = (await reader.read_indicator_broker_evidence("dsn", ["R_25"]))[("R_25", 300, "pullback-1-fixed")]
    assert sample.purchased == sample.settled == 200
