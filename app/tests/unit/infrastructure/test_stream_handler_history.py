from unittest.mock import AsyncMock

import pytest

from src.domain.models.market_data import Candle
from src.infrastructure.handlers.stream_handler import StreamHandler


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"candles": []})
    return ws


@pytest.mark.asyncio
async def test_fetch_symbol_history_paginates(mock_ws):
    mock_ws.is_running = True
    page_one = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 2000 + i} for i in range(3)]
    page_two = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.02, "epoch": 1000 + i} for i in range(2)]

    async def send_side_effect(req):
        if req.get("end") == "latest":
            return {"candles": page_one}
        return {"candles": page_two}

    mock_ws.send = AsyncMock(side_effect=send_side_effect)
    sh = StreamHandler(
        mock_ws,
        ["OTC_SPC"],
        {"fetch_count": 5, "history_fetch_chunk": 3, "history_fetch_delay_seconds": 0, "granularity": 900},
    )
    await sh._fetch_symbol_history("OTC_SPC", 5, granularity=900, store=sh.macro_candles)
    assert len(sh.macro_candles["OTC_SPC"]) == 5
    assert sh.macro_candles["OTC_SPC"][0].epoch == 1000
    assert sh.macro_candles["OTC_SPC"][-1].epoch == 2002


@pytest.mark.asyncio
async def test_fetch_symbol_history_trims_excess(mock_ws):
    mock_ws.is_running = True
    candles = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 1000 + i} for i in range(6)]
    mock_ws.send = AsyncMock(return_value={"candles": candles})
    sh = StreamHandler(
        mock_ws, ["OTC_SPC"], {"history_fetch_chunk": 10, "history_fetch_delay_seconds": 0, "granularity": 900}
    )
    await sh._fetch_symbol_history("OTC_SPC", 4, granularity=900, store=sh.macro_candles)
    assert len(sh.macro_candles["OTC_SPC"]) == 4
    assert sh.macro_candles["OTC_SPC"][0].epoch == 1002
    assert sh.macro_candles["OTC_SPC"][-1].epoch == 1005


@pytest.mark.asyncio
async def test_fetch_symbol_history_stops_on_api_error(mock_ws):
    mock_ws.send = AsyncMock(return_value={"error": {"message": "Invalid granularity"}})
    sh = StreamHandler(mock_ws, ["OTC_SPC"], {"granularity": 60, "history_fetch_delay_seconds": 0})
    await sh._fetch_symbol_history("OTC_SPC", 10, granularity=60, store=sh.micro_candles)
    assert sh.micro_candles["OTC_SPC"] == []


@pytest.mark.asyncio
async def test_ensure_cluster_history_backfills_and_skips_full(mock_ws):
    mock_ws.is_running = True
    page = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 1000 + i} for i in range(3)]
    mock_ws.send = AsyncMock(return_value={"candles": page})
    sh = StreamHandler(
        mock_ws,
        ["OTC_SPC", "R_50"],
        {
            "granularity": 900,
            "history_fetch_chunk": 10,
            "history_fetch_delay_seconds": 0,
            "history_fetch_symbol_delay_seconds": 0,
        },
    )
    sh.macro_candles["OTC_SPC"] = [Candle("OTC_SPC", 1, 1, 1, 1, None, 5000) for _ in range(4)]
    await sh.ensure_cluster_history(3)
    assert len(sh.macro_candles["OTC_SPC"]) == 4
    assert len(sh.macro_candles["R_50"]) == 3


@pytest.mark.asyncio
async def test_ensure_cluster_history_micro_timeframe(mock_ws):
    mock_ws.is_running = True
    page = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 2000 + i} for i in range(5)]
    mock_ws.send = AsyncMock(return_value={"candles": page})
    sh = StreamHandler(
        mock_ws,
        ["OTC_SPC"],
        {
            "granularity": 3600,
            "micro_granularity": 900,
            "history_fetch_chunk": 10,
            "history_fetch_delay_seconds": 0,
            "history_fetch_symbol_delay_seconds": 0,
        },
    )
    await sh.ensure_cluster_history(5, timeframe="micro")
    assert len(sh.micro_candles["OTC_SPC"]) == 5
    assert len(sh.macro_candles["OTC_SPC"]) == 0
