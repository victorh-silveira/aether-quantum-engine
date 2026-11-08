"""Testes unitarios para formatadores de telemetria de movimento e vela ao vivo."""

from src.application.services.market_audit_movement import (
    format_live_candle_line,
    format_next_movement_line,
)
from src.application.services.market_forming_candle import LiveCandleSnapshot


def test_format_live_candle_line():
    snap = LiveCandleSnapshot(
        symbol="1HZ75V",
        epoch=1000,
        open_price=100.0,
        high_price=105.0,
        low_price=99.0,
        spot_price=104.0,
        elapsed_seconds=150,
        remaining_seconds=150,
        progress_pct=0.5,
        current_side="CALL",
        body_span=4.0,
        upper_wick_ratio=0.1667,
        lower_wick_ratio=0.1667,
        tick_velocity=0.5,
        tick_acceleration=0.1,
    )
    line = format_live_candle_line(snap, boundary_seconds=300)
    assert "[LIVE_CANDLE] || M5 || 1HZ75V: CALL (+4.00000)" in line
    assert "150s/300s (50%)" in line
    assert "spot=104.00000" in line
    assert "wicks: U=17% L=17%" in line


def test_format_next_movement_line():
    metrics = {
        "predicted_movement_delta": 0.0035,
        "predicted_movement_side": "CALL",
        "movement_confluence": True,
        "movement_atr_ratio": 1.45,
    }
    line = format_next_movement_line("1HZ75V", metrics)
    assert "[NEXT_MOVE] || 1HZ75V || EXP_DELTA: +0.3500%" in line
    assert "SIDE: CALL" in line
    assert "CONFLUENCE: ALIGNED" in line
    assert "ATR_RATIO: 1.45x" in line


def test_format_next_movement_line_empty_or_invalid():
    assert format_next_movement_line("1HZ75V", {}) == "[NEXT_MOVE] || 1HZ75V || N/A"
    assert (
        format_next_movement_line("1HZ75V", {"predicted_movement_delta": "invalid"}) == "[NEXT_MOVE] || 1HZ75V || N/A"
    )
    assert (
        format_next_movement_line("1HZ75V", {"predicted_movement_delta": float("nan")})
        == "[NEXT_MOVE] || 1HZ75V || N/A"
    )
