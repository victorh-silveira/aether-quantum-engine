"""Testes unitarios para rastreamento de vela ao-vivo (market_forming_candle)."""

from datetime import datetime
from types import SimpleNamespace

from src.application.services.market_forming_candle import (
    LiveCandleSnapshot,
    _extract_forming_candle_from_stream,
    _safe_float,
    build_live_forming_candle_snapshot,
)
from src.domain.models.market_data import Candle


def test_safe_float_conversion():
    assert _safe_float(12.34) == 12.34
    assert _safe_float("12.34") == 12.34
    assert _safe_float("bad", default=5.0) == 5.0
    assert _safe_float(float("nan"), default=1.0) == 1.0


def test_extract_forming_candle_from_stream_empty_or_none():
    assert _extract_forming_candle_from_stream(None, "1HZ75V") is None
    assert _extract_forming_candle_from_stream(SimpleNamespace(micro_candles=None), "1HZ75V") is None
    stream = SimpleNamespace(micro_candles={})
    assert _extract_forming_candle_from_stream(stream, "1HZ75V") is None
    stream.micro_candles["1HZ75V"] = []
    assert _extract_forming_candle_from_stream(stream, "1HZ75V") is None


def test_build_live_forming_candle_snapshot_none_orch_or_symbol():
    assert build_live_forming_candle_snapshot(None, "1HZ75V") is None
    assert build_live_forming_candle_snapshot(SimpleNamespace(), "") is None
    orch = SimpleNamespace(stream=None)
    assert build_live_forming_candle_snapshot(orch, "1HZ75V") is None


def test_build_live_forming_candle_snapshot_call_forming():
    candle = Candle(
        symbol="1HZ75V",
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        time=datetime.now(),
        epoch=1000,
    )
    buffer = SimpleNamespace(
        latest_price=lambda s: 104.0,
        forming_bar_micro_stats=lambda s: SimpleNamespace(price_velocity=0.5, price_acceleration=0.1),
    )
    stream = SimpleNamespace(micro_candles={"1HZ75V": [candle]}, tick_buffer=buffer)
    orch = SimpleNamespace(stream=stream)

    snap = build_live_forming_candle_snapshot(orch, "1HZ75V", boundary_seconds=300, current_timestamp=1150)
    assert snap is not None
    assert isinstance(snap, LiveCandleSnapshot)
    assert snap.symbol == "1HZ75V"
    assert snap.open_price == 100.0
    assert snap.spot_price == 104.0
    assert snap.high_price == 105.0
    assert snap.low_price == 99.0
    assert snap.current_side == "CALL"
    assert snap.elapsed_seconds == 150
    assert snap.remaining_seconds == 150
    assert snap.progress_pct == 0.5
    assert snap.tick_velocity == 0.5
    assert snap.tick_acceleration == 0.1
    assert abs(snap.upper_wick_ratio - (1.0 / 6.0)) < 1e-6
    assert abs(snap.lower_wick_ratio - (1.0 / 6.0)) < 1e-6


def test_build_live_forming_candle_snapshot_put_forming():
    candle = Candle(
        symbol="1HZ75V",
        open=100.0,
        high=101.0,
        low=95.0,
        close=97.0,
        time=datetime.now(),
        epoch=1000,
    )
    stream = SimpleNamespace(micro_candles={"1HZ75V": [candle]}, tick_buffer=None)
    orch = SimpleNamespace(stream=stream)

    snap = build_live_forming_candle_snapshot(orch, "1HZ75V", boundary_seconds=300, current_timestamp=1300)
    assert snap is not None
    assert snap.current_side == "PUT"
    assert snap.spot_price == 97.0
    assert snap.elapsed_seconds == 300
    assert snap.remaining_seconds == 0
    assert snap.progress_pct == 1.0


def test_build_live_forming_candle_snapshot_doji():
    candle = Candle(
        symbol="1HZ75V",
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        time=datetime.now(),
        epoch=1000,
    )
    stream = SimpleNamespace(micro_candles={"1HZ75V": [candle]}, tick_buffer=None)
    orch = SimpleNamespace(stream=stream)

    snap = build_live_forming_candle_snapshot(orch, "1HZ75V", boundary_seconds=300, current_timestamp=1000)
    assert snap is not None
    assert snap.current_side == "DOJI"
    assert snap.upper_wick_ratio == 0.0
    assert snap.lower_wick_ratio == 0.0
