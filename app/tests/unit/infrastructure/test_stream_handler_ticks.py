from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.handlers.stream_handler import StreamHandler


@pytest.fixture
def mock_ws():
    ws = MagicMock()
    ws.send = AsyncMock(
        return_value={"candles": [{"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "epoch": 1600000000}]}
    )
    return ws


@pytest.fixture
def stream_handler(mock_ws):
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 60}
    return StreamHandler(mock_ws, ["R_50"], config)


@pytest.mark.asyncio
async def test_stream_handler_on_tick_records_valid_tick(stream_handler):
    await stream_handler._on_tick({"tick": {"symbol": "R_50", "epoch": 1600000001, "quote": 100.25}})
    ticks = stream_handler.tick_buffer._live["R_50"]
    assert len(ticks) == 1
    assert ticks[0] == (1600000001000, 100.25)


@pytest.mark.asyncio
async def test_stream_handler_on_tick_ignores_invalid_payload(stream_handler):
    await stream_handler._on_tick({"tick": "invalid"})
    await stream_handler._on_tick({"tick": {"symbol": "UNKNOWN", "epoch": 1, "quote": 1.0}})
    await stream_handler._on_tick({"tick": {"symbol": "R_50", "epoch": None, "quote": 1.0}})
    await stream_handler._on_tick({"tick": {"symbol": "R_50", "epoch": 1, "quote": None}})
    await stream_handler._on_tick({"tick": {"symbol": "R_50", "epoch": "bad", "quote": 1.0}})
    assert len(stream_handler.tick_buffer._live["R_50"]) == 0
