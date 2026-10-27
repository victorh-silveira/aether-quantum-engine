"""Testes unitarios para o servico de quarentena de perda alinhada a velas M15."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.settlement_loss_quarantine import (
    calculate_loss_cooldown_expiry,
    handle_settlement_loss_quarantine,
    resolve_loss_symbol,
)


def test_resolve_loss_symbol_cases():
    assert resolve_loss_symbol({"symbol": "OTC_AEX"}, None, None, 1) == "OTC_AEX"
    contract_mock = MagicMock(symbol="OTC_DJI")
    assert resolve_loss_symbol(None, contract_mock, None, 1) == "OTC_DJI"
    orch_mock = MagicMock()
    orch_mock.risk_manager.contract_to_symbol = {123: "OTC_GDAXI"}
    assert resolve_loss_symbol(None, None, orch_mock, 123) == "OTC_GDAXI"
    assert resolve_loss_symbol(None, None, orch_mock, 999) is None


def test_calculate_loss_cooldown_expiry_disabled():
    orch = MagicMock(config={"pullback_strategy": {"enabled": False}})
    expiry, ttl = calculate_loss_cooldown_expiry(orch, "OTC_SSMI", 1000.0)
    assert expiry == 1900.0
    assert ttl == 900


def test_calculate_loss_cooldown_expiry_enabled():
    orch = MagicMock(
        config={
            "pullback_strategy": {
                "enabled": True,
                "period_seconds": 900,
                "loss_cooldown_candles": 1,
                "consecutive_loss_cooldown_candles": 2,
            }
        },
        _symbol_consecutive_losses={"OTC_SSMI": 1},
    )
    now_ts = 905.0
    expiry, ttl = calculate_loss_cooldown_expiry(orch, "OTC_SSMI", now_ts)
    assert expiry == 1800.0 - 5.0
    assert ttl == int(1795.0 - 905.0)

    orch._symbol_consecutive_losses["OTC_SSMI"] = 2
    expiry_2, ttl_2 = calculate_loss_cooldown_expiry(orch, "OTC_SSMI", now_ts)
    assert expiry_2 == 1800.0 + 900.0 - 5.0
    assert ttl_2 > ttl


def test_calculate_loss_cooldown_expiry_zero_candles():
    orch = MagicMock(
        config={
            "pullback_strategy": {
                "enabled": True,
                "period_seconds": 900,
                "loss_cooldown_candles": 0,
            }
        },
        _symbol_consecutive_losses={"OTC_SSMI": 1},
    )
    expiry, ttl = calculate_loss_cooldown_expiry(orch, "OTC_SSMI", 905.0)
    assert expiry == 0.0
    assert ttl == 0


def test_calculate_loss_cooldown_expiry_exact_boundary():
    orch = MagicMock(
        config={
            "pullback_strategy": {
                "enabled": True,
                "period_seconds": 900,
                "loss_cooldown_candles": 1,
            }
        },
        _symbol_consecutive_losses={"OTC_SSMI": 1},
    )
    expiry, ttl = calculate_loss_cooldown_expiry(orch, "OTC_SSMI", 900.0)
    assert expiry == 1800.0 - 5.0
    assert ttl == int(1795.0 - 900.0)


@pytest.mark.asyncio
async def test_handle_settlement_loss_quarantine_profit_resets():
    orch = MagicMock(_symbol_consecutive_losses={"OTC_SSMI": 2})
    await handle_settlement_loss_quarantine(orch, {"symbol": "OTC_SSMI"}, None, 1, 10.0)
    assert orch._symbol_consecutive_losses["OTC_SSMI"] == 0


@pytest.mark.asyncio
async def test_handle_settlement_loss_quarantine_no_symbol():
    orch = MagicMock(risk_manager=MagicMock(contract_to_symbol={}))
    orch._symbol_loss_quarantine = {}
    await handle_settlement_loss_quarantine(orch, None, None, 1, -10.0)
    assert orch._symbol_loss_quarantine == {}


@pytest.mark.asyncio
async def test_handle_settlement_loss_quarantine_loss_applies():
    redis_mock = AsyncMock()
    orch = MagicMock(
        config={
            "trading": {"mode": "demo"},
            "pullback_strategy": {"enabled": True, "period_seconds": 900, "loss_cooldown_candles": 1},
        },
        _symbol_consecutive_losses={},
        state_store=MagicMock(redis=redis_mock),
    )
    await handle_settlement_loss_quarantine(orch, {"symbol": "OTC_SSMI"}, None, 1, -10.0)
    assert orch._symbol_consecutive_losses["OTC_SSMI"] == 1
    assert "OTC_SSMI" in orch._symbol_loss_quarantine
    redis_mock.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_settlement_loss_quarantine_redis_exception():
    redis_mock = AsyncMock()
    redis_mock.set.side_effect = RuntimeError("redis down")
    orch = MagicMock(
        config={
            "trading": {"mode": "demo"},
            "pullback_strategy": {"enabled": True, "period_seconds": 900, "loss_cooldown_candles": 1},
        },
        _symbol_consecutive_losses={},
        state_store=MagicMock(redis=redis_mock),
    )
    await handle_settlement_loss_quarantine(orch, {"symbol": "OTC_SSMI"}, None, 1, -10.0)
    assert orch._symbol_consecutive_losses["OTC_SSMI"] == 1
