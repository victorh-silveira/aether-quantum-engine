from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models.market_data import Candle
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
    symbols = ["R_50", "R_50"]
    return StreamHandler(mock_ws, symbols, config)


def test_resolve_fetch_count_explicit(mock_ws):
    sh = StreamHandler(mock_ws, ["R_50"], {"fetch_count": 320})
    assert sh._resolve_fetch_count() == 320


def test_resolve_fetch_count_from_history_bars(mock_ws):
    sh = StreamHandler(mock_ws, ["R_50"], {"history_bars": 288, "history_warmup_bars": 32})
    assert sh._resolve_fetch_count() == 320


def test_resolve_fetch_count_default(mock_ws):
    sh = StreamHandler(mock_ws, ["R_50"], {"granularity": 300})
    assert sh._resolve_fetch_count() == 500


def test_resolve_fetch_count_startup_override(mock_ws):
    sh = StreamHandler(mock_ws, ["R_50"], {"fetch_count": 25984, "_startup_fetch_count": 192})
    assert sh._resolve_fetch_count() == 192
    sh.config.pop("_startup_fetch_count", None)
    assert sh._resolve_fetch_count() == 25984


def test_stream_handler_normalizes_unsupported_granularity(mock_ws):
    sh = StreamHandler(mock_ws, ["R_50"], {"granularity": 10})
    assert sh.granularity == 60


@pytest.mark.asyncio
async def test_stream_handler_on_candle(stream_handler):
    candle_data = {
        "ohlc": {"symbol": "R_50", "open": 1.4, "high": 1.5, "low": 1.3, "close": 1.45, "open_time": 1600000000}
    }
    await stream_handler._on_candle(candle_data)
    assert stream_handler.candles["R_50"][-1].close == 1.45


def test_stream_handler_get_numpy(stream_handler):
    stream_handler.candles["R_50"] = [Candle("R_50", 1.0, 1.1, 0.9, 1.05, datetime.now(), 1000)]
    series = stream_handler.get_numpy_series("R_50")
    assert series.tolist() == [1.05]


@pytest.mark.asyncio
async def test_stream_handler_start_stream(stream_handler, mock_ws):
    callback = AsyncMock()
    await stream_handler.start_candle_stream(callback)
    assert mock_ws.subscribe.called
    assert mock_ws.send.called


def test_stream_handler_unknown_symbol(stream_handler):
    assert len(stream_handler.get_numpy_series("UNKNOWN")) == 0


def test_get_last_candle_epoch(stream_handler):
    assert stream_handler.get_last_candle_epoch("UNKNOWN") is None
    stream_handler.candles["R_50"] = [Candle("R_50", 1.0, 1.1, 0.9, 1.05, datetime.now(), 1600000123)]
    assert stream_handler.get_last_candle_epoch("R_50") == 1600000123


@pytest.mark.asyncio
async def test_stream_handler_candle_error(stream_handler):
    await stream_handler._on_candle({"invalid": "data"})
    assert len(stream_handler.candles["R_50"]) == 0

    await stream_handler._on_candle({"ohlc": {"symbol": "UNKNOWN", "open": 1.0, "open_time": 1000}})
    assert len(stream_handler.candles["R_50"]) == 0


@pytest.mark.asyncio
async def test_stream_handler_candle_logic():
    ws = MagicMock()
    ws.send = AsyncMock(return_value={"candles": []})
    sh = StreamHandler(ws, ["R_50"], {"buffer_limit": 2})
    callback = AsyncMock()
    await sh.start_candle_stream(callback)

    await sh._on_candle(
        {"ohlc": {"symbol": "R_50", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "open_time": 1000}}
    )
    await sh._on_candle(
        {"ohlc": {"symbol": "R_50", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.1, "open_time": 1000}}
    )
    await sh._on_candle(
        {"ohlc": {"symbol": "R_50", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.1, "open_time": 1060}}
    )
    await sh._on_candle(
        {"ohlc": {"symbol": "R_50", "open": 1.2, "high": 1.3, "low": 1.1, "close": 1.2, "open_time": 1120}}
    )

    assert len(sh.candles["R_50"]) == 2
    assert callback.called


@pytest.mark.asyncio
async def test_stream_handler_start_stream_fails_when_ws_disconnected(mock_ws):
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 60}
    symbols = ["R_50"]
    sh = StreamHandler(mock_ws, symbols, config)
    mock_ws.is_running = False
    with pytest.raises(ConnectionError):
        await sh.start_candle_stream(AsyncMock())
    assert sh.is_synchronized is False


@pytest.mark.asyncio
async def test_stream_handler_start_stream_fails_after_history_sync(mock_ws):
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 60}
    symbols = ["R_50"]
    sh = StreamHandler(mock_ws, symbols, config)
    mock_ws.is_running = True

    async def drop_after_history(*_args, **_kwargs):
        mock_ws.is_running = False
        return {"candles": [{"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "epoch": 1600000000}]}

    mock_ws.send = AsyncMock(side_effect=drop_after_history)
    with pytest.raises(ConnectionError):
        await sh.start_candle_stream(AsyncMock())
    assert sh.is_synchronized is False


@pytest.mark.asyncio
async def test_fetch_candle_closes_returns_closes(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(
        return_value={"candles": [{"close": "100.1"}, {"close": "100.2"}, {"open": 1, "close": 100.3}]}
    )
    sh = StreamHandler(mock_ws, ["R_50"], {"buffer_limit": 10})
    closes = await sh.fetch_candle_closes("R_50", 300, 5)
    assert closes == [100.1, 100.2, 100.3]
    mock_ws.send.assert_awaited_once()
    req = mock_ws.send.await_args.args[0]
    assert req["granularity"] == 300
    assert req["style"] == "candles"


@pytest.mark.asyncio
async def test_fetch_candle_closes_unknown_symbol(mock_ws):
    mock_ws.is_running = True
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_closes("OTHER", 300, 5) == []


@pytest.mark.asyncio
async def test_fetch_candle_closes_zero_or_ws_down(mock_ws):
    mock_ws.is_running = False
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_closes("R_50", 60, 3) == []
    assert await sh.fetch_candle_closes("R_50", 60, 0) == []


@pytest.mark.asyncio
async def test_fetch_candle_closes_api_error(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(return_value={"error": {"code": "x"}})
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_closes("R_50", 300, 2) == []


@pytest.mark.asyncio
async def test_fetch_candle_closes_send_raises(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(side_effect=RuntimeError("x"))
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_closes("R_50", 300, 2) == []


@pytest.mark.asyncio
async def test_fetch_candle_closes_skips_invalid_rows(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(return_value={"candles": [{"close": "10"}, {"invalid": True}, {"close": "not-float"}]})
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_closes("R_50", 300, 10) == [10.0]


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_returns_tuples(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(
        return_value={
            "candles": [
                {"open": "1", "high": "2", "low": "0.5", "close": "1.5"},
                {"open": 2, "high": 3, "low": 1, "close": 2.5},
            ]
        }
    )
    sh = StreamHandler(mock_ws, ["R_50"], {"buffer_limit": 10})
    rows = await sh.fetch_candle_ohlc("R_50", 300, 5)
    assert rows == [(1.0, 2.0, 0.5, 1.5), (2.0, 3.0, 1.0, 2.5)]
    req = mock_ws.send.await_args.args[0]
    assert req["granularity"] == 300
    assert req["style"] == "candles"


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_unknown_symbol(mock_ws):
    mock_ws.is_running = True
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_ohlc("OTHER", 300, 5) == []


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_zero_or_ws_down(mock_ws):
    mock_ws.is_running = False
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_ohlc("R_50", 60, 3) == []
    assert await sh.fetch_candle_ohlc("R_50", 60, 0) == []


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_api_error(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(return_value={"error": {"code": "x"}})
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_ohlc("R_50", 300, 2) == []


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_send_raises(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(side_effect=RuntimeError("x"))
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_ohlc("R_50", 300, 2) == []


@pytest.mark.asyncio
async def test_fetch_candle_ohlc_skips_invalid_rows(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(
        return_value={
            "candles": [
                {"open": "1", "high": "2", "low": "1", "close": "1"},
                {"invalid": True},
                {"open": "x", "high": "1", "low": "1", "close": "1"},
            ]
        }
    )
    sh = StreamHandler(mock_ws, ["R_50"], {})
    assert await sh.fetch_candle_ohlc("R_50", 300, 10) == [(1.0, 2.0, 1.0, 1.0)]
