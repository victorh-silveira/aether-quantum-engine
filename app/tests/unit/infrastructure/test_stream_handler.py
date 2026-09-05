from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 900, "micro_granularity": 300}
    return StreamHandler(mock_ws, ["R_10"], config)


def test_resolve_fetch_count_explicit(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"fetch_count": 320})
    assert sh._resolve_fetch_count() == 320


def test_resolve_fetch_count_from_history_bars(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"history_bars": 288, "history_warmup_bars": 32})
    assert sh._resolve_fetch_count() == 320


def test_resolve_fetch_count_default(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"granularity": 900})
    assert sh._resolve_fetch_count() == 500


def test_resolve_fetch_count_startup_override(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"fetch_count": 15616, "_startup_fetch_count": 192})
    assert sh._resolve_fetch_count() == 192
    sh.config.pop("_startup_fetch_count", None)
    assert sh._resolve_fetch_count() == 15616


def test_history_sync_quiet_for_small_fetch_goal(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"fetch_count": 256})
    assert sh._history_sync_quiet(256) is True
    assert sh._history_sync_quiet(1000) is False
    sh.config["_startup_fetch_count"] = 5000
    assert sh._history_sync_quiet(5000) is False
    sh.config["_startup_quiet"] = True
    assert sh._history_sync_quiet(5000) is True


def test_resolve_sync_targets_train_lean_skips_mini(mock_ws):
    from src.infrastructure.handlers.stream_sync_start import _resolve_sync_targets

    sh = StreamHandler(
        mock_ws,
        ["R_10"],
        {
            "fetch_count": 2000,
            "micro_fetch_count": 2000,
            "mini_fetch_count": 256,
            "_startup_fetch_count": 2000,
            "_startup_train_lean": True,
            "granularity": 300,
            "micro_granularity": 300,
        },
    )
    macro, micro, mini = _resolve_sync_targets(sh)
    assert macro == 128
    assert micro == 2000
    assert mini == 0


def test_resolve_sync_targets_train_lean_d1_uses_365(mock_ws):
    from src.infrastructure.handlers.stream_sync_start import _resolve_sync_targets

    sh = StreamHandler(
        mock_ws,
        ["1HZ75V"],
        {
            "fetch_count": 500,
            "micro_fetch_count": 500,
            "history_bars": 500,
            "_startup_fetch_count": 500,
            "_startup_train_lean": True,
            "granularity": 86400,
            "micro_granularity": 300,
        },
    )
    macro, micro, mini = _resolve_sync_targets(sh)
    assert macro == 365
    assert micro == 500
    assert mini == 0


def test_resolve_sync_targets_live_keeps_triple(mock_ws):
    from src.infrastructure.handlers.stream_sync_start import _resolve_sync_targets

    sh = StreamHandler(
        mock_ws,
        ["R_10"],
        {
            "fetch_count": 2000,
            "micro_fetch_count": 2000,
            "mini_fetch_count": 256,
        },
    )
    macro, micro, mini = _resolve_sync_targets(sh)
    assert macro == 2000
    assert micro == 2000
    assert mini == 256


def test_stream_handler_normalizes_unsupported_granularity(mock_ws):
    sh = StreamHandler(mock_ws, ["R_10"], {"granularity": 10})
    assert sh.macro_granularity == 60


def test_stream_handler_get_numpy(stream_handler):
    stream_handler.macro_candles["R_10"] = [Candle("R_10", 1.0, 1.1, 0.9, 1.05, datetime.now(), 1000)]
    series = stream_handler.get_numpy_series("R_10")
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
    stream_handler.macro_candles["R_10"] = [Candle("R_10", 1.0, 1.1, 0.9, 1.05, datetime.now(), 1600000123)]
    assert stream_handler.get_last_candle_epoch("R_10") == 1600000123


@pytest.mark.asyncio
async def test_stream_handler_candle_error(stream_handler):
    await stream_handler._on_candle({"invalid": "data"})
    assert len(stream_handler.macro_candles["R_10"]) == 0


@pytest.mark.asyncio
async def test_stream_handler_start_stream_fails_when_ws_disconnected(mock_ws):
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 900, "micro_granularity": 300}
    sh = StreamHandler(mock_ws, ["R_10"], config)
    mock_ws.is_running = False
    with pytest.raises(ConnectionError):
        await sh.start_candle_stream(AsyncMock())
    assert sh.is_synchronized is False


@pytest.mark.asyncio
async def test_stream_handler_start_stream_fails_after_history_sync(mock_ws):
    config = {"buffer_limit": 10, "fetch_count": 3, "granularity": 900, "micro_granularity": 300}
    sh = StreamHandler(mock_ws, ["R_10"], config)
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
    sh = StreamHandler(mock_ws, ["R_10"], {"buffer_limit": 10})
    closes = await sh.fetch_candle_closes("R_10", 900, 5)
    assert closes == [100.1, 100.2, 100.3]
    req = mock_ws.send.await_args.args[0]
    assert req["granularity"] == 900


@pytest.mark.asyncio
async def test_fetch_candle_closes_unknown_symbol(mock_ws):
    mock_ws.is_running = True
    sh = StreamHandler(mock_ws, ["R_10"], {})
    assert await sh.fetch_candle_closes("OTHER", 900, 5) == []
    assert await sh.fetch_candle_ohlc("OTHER", 900, 5) == []


@pytest.mark.asyncio
async def test_fetch_candle_closes_skips_invalid(mock_ws):
    mock_ws.is_running = True
    mock_ws.send = AsyncMock(return_value={"candles": [{"close": "1.5"}, {"invalid": 1}]})
    sh = StreamHandler(mock_ws, ["R_10"], {})
    assert await sh.fetch_candle_closes("R_10", 900, 2) == [1.5]


@pytest.mark.asyncio
async def test_fetch_candle_closes_zero_or_ws_down(mock_ws):
    mock_ws.is_running = False
    sh = StreamHandler(mock_ws, ["R_10"], {})
    assert await sh.fetch_candle_closes("R_10", 60, 3) == []
    assert await sh.fetch_candle_closes("R_10", 60, 0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_or_err", [RuntimeError("x"), {"error": {"code": "x"}}])
async def test_fetch_candle_failures(mock_ws, exc_or_err):
    mock_ws.is_running = True
    if isinstance(exc_or_err, Exception):
        mock_ws.send = AsyncMock(side_effect=exc_or_err)
    else:
        mock_ws.send = AsyncMock(return_value=exc_or_err)
    sh = StreamHandler(mock_ws, ["R_10"], {})
    assert await sh.fetch_candle_closes("R_10", 900, 2) == []
    assert await sh.fetch_candle_ohlc("R_10", 900, 2) == []


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
    sh = StreamHandler(mock_ws, ["R_10"], {})
    assert await sh.fetch_candle_ohlc("R_10", 900, 10) == [(1.0, 2.0, 1.0, 1.0)]


@pytest.mark.asyncio
async def test_stream_handler_reconnect_stream(stream_handler):
    orch = MagicMock()
    with patch(
        "src.infrastructure.handlers.stream_handler.execute_stream_reconnect",
        new_callable=AsyncMock,
        return_value=True,
    ) as reconnect_mock:
        ok = await stream_handler.reconnect_stream(orch)
    assert ok is True
    reconnect_mock.assert_awaited_once_with(orch, stream_handler)


@pytest.mark.asyncio
async def test_stream_handler_reconnect_single_flight(stream_handler):
    orch = MagicMock()
    stream_handler._reconnect_in_progress = True
    ok = await stream_handler.reconnect_stream(orch)
    assert ok is False


@pytest.mark.asyncio
async def test_stream_handler_equal_micro_mini_dispatch(mock_ws):
    config = {
        "buffer_limit": 10,
        "fetch_count": 3,
        "granularity": 86400,
        "micro_granularity": 300,
        "mini_granularity": 300,
    }
    sh = StreamHandler(mock_ws, ["R_10"], config)
    sh.micro_candles["R_10"] = []
    sh.mini_candles["R_10"] = []
    payload = {
        "ohlc": {
            "symbol": "R_10",
            "granularity": 300,
            "open": "1.0",
            "high": "1.2",
            "low": "0.9",
            "close": "1.1",
            "open_time": 1600000000,
            "epoch": 1600000000,
        }
    }
    await sh._on_candle(payload)
    assert len(sh.micro_candles["R_10"]) == 1
    assert len(sh.mini_candles["R_10"]) == 1
    series = sh.get_mini_numpy_series("R_10", "close")
    assert series.tolist() == [1.1]

    # Test fallback to micro_candles when mini_candles is empty
    sh.mini_candles["R_10"] = []
    fallback_series = sh.get_mini_numpy_series("R_10", "close")
    assert fallback_series.tolist() == [1.1]
