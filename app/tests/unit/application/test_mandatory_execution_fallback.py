from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.mandatory_execution_fallback import (
    _resolve_fallback_direction,
    _select_fallback_symbol,
    resolve_mandatory_fallback_order,
)
from src.domain.models.trade import TradeDirection


def test_select_fallback_symbol_branches():
    orch = MagicMock()
    orch.symbols = ["OTC_SPC", "OTC_NDX"]

    # Caso 1: Anchor OTC_SSMI aberto
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.services.orchestrator.mandatory_execution_fallback.is_otc_exchange_open_brt",
            lambda s, ts=None: s == "OTC_SSMI",
        )
        assert _select_fallback_symbol(orch) == "OTC_SSMI"

    # Caso 2: Anchor fechado, primeiro ativo de symbols aberto
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.services.orchestrator.mandatory_execution_fallback.is_otc_exchange_open_brt",
            lambda s, ts=None: s == "OTC_SPC",
        )
        assert _select_fallback_symbol(orch) == "OTC_SPC"

    # Caso 3: Nenhum ativo aberto, fallback para anchor
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.services.orchestrator.mandatory_execution_fallback.is_otc_exchange_open_brt",
            lambda s, ts=None: False,
        )
        assert _select_fallback_symbol(orch) == "OTC_SSMI"


def test_resolve_fallback_direction_branches():
    orch = MagicMock()
    orch.config = {"pullback_strategy": {"period_seconds": 900}}

    # Caso 1: Sinal existente no indicador
    sig = MagicMock()
    sig.side = "PUT"
    orch._indicator_period_signals = {("OTC_SSMI", 900): sig}
    direction, reason, flipped = _resolve_fallback_direction(orch, "OTC_SSMI")
    assert direction == TradeDirection.PUT
    assert not flipped

    # Caso 2: Sem sinal, buffer com close >= open (CALL)
    orch._indicator_period_signals = {}
    bar_bull = MagicMock(close=100.5, open=100.0)
    orch.data_buffer = {"OTC_SSMI": [bar_bull]}
    direction, reason, flipped = _resolve_fallback_direction(orch, "OTC_SSMI")
    assert direction == TradeDirection.CALL
    assert not flipped

    # Caso 3: Sem sinal, buffer com close < open (PUT)
    bar_bear = MagicMock(close=99.5, open=100.0)
    orch.data_buffer = {"OTC_SSMI": [bar_bear]}
    direction, reason, flipped = _resolve_fallback_direction(orch, "OTC_SSMI")
    assert direction == TradeDirection.PUT
    assert not flipped

    # Caso 4: Sem buffer
    orch.data_buffer = {}
    direction, reason, flipped = _resolve_fallback_direction(orch, "OTC_SSMI")
    assert direction == TradeDirection.CALL
    assert not flipped

    # Caso 5: Ativo invertido pela telemetria (ex: OTC_HSI)
    orch.config["pullback_strategy"]["telemetry_inverted_symbols"] = ["OTC_HSI"]
    bar_bull = MagicMock(close=100.5, open=100.0)
    orch.data_buffer = {"OTC_HSI": [bar_bull]}
    direction, reason, flipped = _resolve_fallback_direction(orch, "OTC_HSI")
    assert direction == TradeDirection.PUT
    assert flipped
    assert reason == "TELEMETRY_INVERTED_BIAS"


@pytest.mark.asyncio
async def test_resolve_mandatory_fallback_order_balance_zero():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = {"pullback_strategy": {"period_seconds": 900}, "risk_management": {"params": {"stake_min": 1.0}}}
    orch.state.balance = 0.0

    res = await resolve_mandatory_fallback_order(orch)
    assert res == []


@pytest.mark.asyncio
async def test_resolve_mandatory_fallback_order_api_exception():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = {"pullback_strategy": {"period_seconds": 900}, "risk_management": {"params": {"stake_min": 1.0}}}
    orch.state.balance = 1000.0
    orch.trade_handler.fetch_proposal_snapshot = AsyncMock(side_effect=RuntimeError("WSS timeout"))

    res = await resolve_mandatory_fallback_order(orch)
    assert res == []


@pytest.mark.asyncio
async def test_resolve_mandatory_fallback_order_proposal_rejected():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = {"pullback_strategy": {"period_seconds": 900}, "risk_management": {"params": {"stake_min": 1.0}}}
    orch.state.balance = 1000.0
    orch.trade_handler.fetch_proposal_snapshot = AsyncMock(return_value={"error": "Contract unavailable"})

    res = await resolve_mandatory_fallback_order(orch)
    assert res == []


@pytest.mark.asyncio
async def test_resolve_mandatory_fallback_order_success():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = {"pullback_strategy": {"period_seconds": 900}, "risk_management": {"params": {"stake_min": 1.0}}}
    orch.state.balance = 1000.0
    quote = {
        "proposal_id": "prop_12345",
        "price": 10.0,
        "payout": 17.5,
        "rate": 0.75,
    }
    orch.trade_handler.fetch_proposal_snapshot = AsyncMock(return_value=quote)

    res = await resolve_mandatory_fallback_order(orch)
    assert len(res) == 1
    sym, direction, metrics = res[0]
    assert sym == "OTC_SSMI"
    assert direction in (TradeDirection.CALL, TradeDirection.PUT)
    assert metrics["_prequoted_proposal_id"] == "prop_12345"
    assert metrics["organic_decision_reason"].startswith("MANDATORY_")
    assert metrics.get("indicator_controls_execution") is True
    assert metrics.get("direction_origin") == "PULLBACK_SETUP"
    assert "direction_telemetry_flipped" in metrics


@pytest.mark.asyncio
async def test_resolve_mandatory_fallback_order_rotation():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.symbols = ["OTC_SPC", "OTC_NDX"]
    orch.config = {"pullback_strategy": {"period_seconds": 900}, "risk_management": {"params": {"stake_min": 1.0}}}
    orch.state.balance = 1000.0

    async def mock_fetch(sym, direction, stake, params=None):
        if sym == "OTC_SSMI":
            return None
        return {"proposal_id": f"prop_{sym}", "price": 10.0, "payout": 18.0, "rate": 0.80}

    orch.trade_handler.fetch_proposal_snapshot = AsyncMock(side_effect=mock_fetch)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.services.orchestrator.mandatory_execution_fallback.is_otc_exchange_open_brt",
            lambda s, ts=None: True,
        )
        res = await resolve_mandatory_fallback_order(orch)
        assert len(res) == 1
        sym, direction, metrics = res[0]
        assert sym == "OTC_SPC"
        assert metrics["_prequoted_proposal_id"] == "prop_OTC_SPC"
