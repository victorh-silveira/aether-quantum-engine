"""Cotacoes observadas antes da compra e separacao das coortes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.handlers.trade_handler_quote_observation import record_indicator_quote


@pytest.mark.asyncio
async def test_confirmatory_quote_is_audited_before_buy():
    writer = SimpleNamespace(record_quote_observation=AsyncMock())
    params = {
        "_indicator_model_version": "v1",
        "_indicator_period_seconds": 900,
        "_signal_candle_epoch": 900,
        "_quote_guard_side_probability": 0.60,
    }
    await record_indicator_quote(writer, {"trading": {"mode": "demo"}}, "R_25", "PUT", params, "p1", 10, 18, 0.8)
    row = writer.record_quote_observation.await_args.args[0]
    assert row["cohort"] == "confirmatory"
    assert row["period_seconds"] == 900
    assert row["quote_ev"] == pytest.approx(0.08)
    assert row["proposal_id"] == "p1"


@pytest.mark.asyncio
async def test_exploratory_quote_and_missing_audit_writer():
    writer = SimpleNamespace(record_quote_observation=AsyncMock())
    params = {"_indicator_model_version": "v2", "_quote_guard_side_probability": 0.52}
    await record_indicator_quote(writer, {}, "R_10", "CALL", params, "p2", 10, 18, 0.8)
    row = writer.record_quote_observation.await_args.args[0]
    assert row["cohort"] == "exploratory"
    assert row["period_seconds"] == 300
    assert row["account_mode"] == "demo"
    await record_indicator_quote(writer, {}, "R_10", "CALL", {}, "p3", 10, 18, 0.8)
    assert writer.record_quote_observation.await_count == 1
    with pytest.raises(RuntimeError, match="indisponivel"):
        await record_indicator_quote(None, {}, "R_10", "CALL", params, "p4", 10, 18, 0.8)
    with pytest.raises(RuntimeError, match="indisponivel"):
        await record_indicator_quote(object(), {}, "R_10", "CALL", params, "p4", 10, 18, 0.8)


@pytest.mark.asyncio
async def test_quote_without_valid_payout_is_exploratory():
    writer = SimpleNamespace(record_quote_observation=AsyncMock())
    await record_indicator_quote(
        writer,
        {},
        "R_25",
        "CALL",
        {"_indicator_model_version": "v3", "_quote_guard_side_probability": None},
        "p3",
        10,
        None,
        None,
    )
    row = writer.record_quote_observation.await_args.args[0]
    assert row["quote_ev"] is None
    assert row["cohort"] == "exploratory"


@pytest.mark.asyncio
async def test_setup_demo_quote_has_no_invented_probability_or_ev():
    writer = SimpleNamespace(record_quote_observation=AsyncMock())
    params = {"_indicator_model_version": "pullback-1-fixed", "_quote_guard_side_probability": 0.5}
    await record_indicator_quote(writer, {"trading": {"mode": "demo"}}, "R_25", "CALL", params, "p1", 1, 1.8, 0.8)
    row = writer.record_quote_observation.await_args.args[0]
    assert row["cohort"] == "setup_pilot"
    assert row["side_probability"] is None
    assert row["quote_ev"] is None
