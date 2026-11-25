"""Atualizacao H1 anterior a avaliacao M15 na virada de hora."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.indicator.domain import Candle
from src.indicator.pullback import SetupResult
from src.indicator.pullback_service import PullbackService


@pytest.mark.asyncio
async def test_h1_is_refreshed_before_m15_and_reused_inside_hour(monkeypatch):
    now = 100 * 3600
    captured = []

    async def candles(symbol, count, period):
        end = now // period * period
        return [
            {"epoch": end - (60 - index) * period, "open": 100, "high": 102, "low": 99, "close": 101}
            for index in range(60)
        ]

    def evaluate(candles, period, observed_at, **kwargs):
        if period == 900:
            captured.append(kwargs["h1_bars"][-1].epoch)
        return SetupResult("SEM SINAL", "diagnostico")

    source = SimpleNamespace(candles=AsyncMock(side_effect=candles))
    sink = SimpleNamespace(promote_setups=AsyncMock(), save_setup=AsyncMock())
    metrics = SimpleNamespace(update=lambda signal: None)
    service = PullbackService(source, sink, metrics, variant="persistent_trend_v3")
    service.symbols = {"OTC_FTSE": "FTSE"}
    service.last_catalog = now
    service._h1_cache["OTC_FTSE"] = [Candle(now - 7200, 100, 102, 99, 101)]
    monkeypatch.setattr("src.indicator.pullback_service.evaluate", evaluate)
    await service.tick(now)
    assert captured == [now - 3600]
    before = sum(call.kwargs["period"] == 3600 for call in source.candles.call_args_list)
    now += 900
    await service.tick(now)
    assert captured[-1] == now // 3600 * 3600 - 3600
    assert sum(call.kwargs["period"] == 3600 for call in source.candles.call_args_list) == before
    now += 2700
    await service.tick(now)
    assert captured[-1] == now - 3600


@pytest.mark.asyncio
async def test_h1_failure_retries_on_next_pending_m15(monkeypatch):
    now = 100 * 3600
    captured = []

    async def candles(symbol, count, period):
        if period == 3600:
            raise OSError("rede indisponivel")
        end = now // period * period
        return [
            {"epoch": end - (60 - index) * period, "open": 100, "high": 102, "low": 99, "close": 101}
            for index in range(60)
        ]

    def evaluate(candles, period, observed_at, **kwargs):
        if period == 900:
            captured.append(kwargs["h1_bars"])
        return SetupResult("SEM SINAL", "diagnostico")

    source = SimpleNamespace(candles=AsyncMock(side_effect=candles))
    sink = SimpleNamespace(promote_setups=AsyncMock(), save_setup=AsyncMock())
    service = PullbackService(source, sink, SimpleNamespace(update=lambda signal: None), variant="persistent_trend_v3")
    service.symbols = {"OTC_FTSE": "FTSE"}
    service.last_catalog = now
    monkeypatch.setattr("src.indicator.pullback_service.evaluate", evaluate)
    service._h1_cache["OTC_FTSE"] = [Candle(now - 7200, 100, 102, 99, 101)]
    await service.tick(now)
    assert captured == [None]
    assert "OTC_FTSE" not in service._h1_cache
    before = sum(call.kwargs["period"] == 3600 for call in source.candles.call_args_list)
    now += 900
    await service.tick(now)
    assert sum(call.kwargs["period"] == 3600 for call in source.candles.call_args_list) > before
