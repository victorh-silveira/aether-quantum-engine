from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models.market_data import Candle
from src.infrastructure.handlers.stream_candle_apply import apply_candle_update, candle_from_ohlc
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.stream_timeframe import (
    ohlc_payload_granularity,
    resolve_dual_granularity,
    resolve_micro_fetch_count,
)


@pytest.fixture
def stream_handler():
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock(return_value={"candles": []})
    return StreamHandler(mock_ws, ["RDBULL"], {"granularity": 900, "micro_granularity": 60})


def test_resolve_dual_granularity_defaults():
    macro, micro = resolve_dual_granularity({"granularity": 900, "micro_granularity": 60})
    assert macro == 900
    assert micro == 60


def test_resolve_micro_fetch_count():
    assert resolve_micro_fetch_count({"micro_fetch_count": 128}) == 128
    assert resolve_micro_fetch_count({"micro_history_bars": 256}) == 256


def test_ohlc_payload_granularity_explicit():
    assert ohlc_payload_granularity({"open_time": 1000, "granularity": 900}, 900, 60) == 900


def test_ohlc_payload_granularity_infers_macro_from_epoch():
    assert ohlc_payload_granularity({"open_time": 1800}, 900, 60) == 900


def test_ohlc_payload_granularity_infers_micro_from_epoch():
    assert ohlc_payload_granularity({"open_time": 120}, 900, 60) == 60


def test_candle_from_ohlc_and_apply_update():
    candle = candle_from_ohlc("RDBULL", {"open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "open_time": 900})
    store: dict[str, list] = {"RDBULL": []}
    last: dict[str, int | None] = {"RDBULL": None}
    result = apply_candle_update(store, last, "RDBULL", candle, limit=5)
    assert result.event == "append"
    assert len(store["RDBULL"]) == 1


@pytest.mark.asyncio
async def test_stream_handler_on_macro_candle(stream_handler):
    await stream_handler._on_candle(
        {
            "ohlc": {
                "symbol": "RDBULL",
                "open": 1.4,
                "high": 1.5,
                "low": 1.3,
                "close": 1.45,
                "open_time": 900,
                "granularity": 900,
            }
        }
    )
    assert stream_handler.macro_candles["RDBULL"][-1].close == 1.45


@pytest.mark.asyncio
async def test_stream_handler_on_micro_candle_invokes_callback(stream_handler):
    callback = AsyncMock()
    stream_handler.candle_callback = callback
    await stream_handler._on_candle(
        {
            "ohlc": {
                "symbol": "RDBULL",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "open_time": 60,
                "granularity": 60,
            }
        }
    )
    callback.assert_awaited_once()
    assert stream_handler.micro_candles["RDBULL"][-1].close == 1.05


@pytest.mark.asyncio
async def test_stream_handler_micro_candle_ignores_unknown_symbol(stream_handler):
    await stream_handler._on_candle(
        {
            "ohlc": {
                "symbol": "UNKNOWN",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "open_time": 60,
                "granularity": 60,
            }
        }
    )
    assert stream_handler.micro_candles["RDBULL"] == []


@pytest.mark.asyncio
async def test_stream_handler_macro_candle_ignores_unknown_symbol(stream_handler):
    await stream_handler._on_candle(
        {
            "ohlc": {
                "symbol": "UNKNOWN",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "open_time": 900,
                "granularity": 900,
            }
        }
    )
    assert stream_handler.macro_candles["RDBULL"] == []


def test_stream_handler_get_micro_numpy(stream_handler):
    stream_handler.micro_candles["RDBULL"] = [Candle("RDBULL", 1.0, 1.1, 0.9, 1.05, datetime.now(), 60)]
    series = stream_handler.get_micro_numpy_series("RDBULL")
    assert series.tolist() == [1.05]


def test_get_last_micro_candle_epoch(stream_handler):
    stream_handler.micro_candles["RDBULL"] = [Candle("RDBULL", 1.0, 1.1, 0.9, 1.05, datetime.now(), 120)]
    assert stream_handler.get_last_micro_candle_epoch("RDBULL") == 120


@pytest.mark.asyncio
async def test_stream_handler_macro_buffer_limit():
    ws = MagicMock()
    ws.send = AsyncMock(return_value={"candles": []})
    sh = StreamHandler(ws, ["RDBULL"], {"buffer_limit": 2, "granularity": 900, "micro_granularity": 60})
    await sh.start_candle_stream(AsyncMock())
    payload = {
        "symbol": "RDBULL",
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
        "granularity": 900,
    }
    await sh._on_candle({"ohlc": {**payload, "open_time": 900}})
    await sh._on_candle({"ohlc": {**payload, "open_time": 900, "close": 1.05}})
    await sh._on_candle({"ohlc": {**payload, "open_time": 1800, "close": 1.1}})
    await sh._on_candle({"ohlc": {**payload, "open_time": 2700, "close": 1.2}})
    assert len(sh.macro_candles["RDBULL"]) == 2
