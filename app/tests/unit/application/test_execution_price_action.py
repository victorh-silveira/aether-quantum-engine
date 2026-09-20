"""Testes unitarios para execution_price_action."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from src.application.services.execution_price_action import (
    _resolve_candle_ohlc,
    should_skip_adverse_tick_flow,
    should_skip_climactic_blowoff,
    should_skip_opposing_marubozu_flow,
    should_skip_wick_rejection,
)
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection


def test_resolve_candle_ohlc_from_metrics():
    metrics = {"closed_candle_ohlc": [10.0, 15.0, 8.0, 12.0]}
    assert _resolve_candle_ohlc(metrics) == (10.0, 15.0, 8.0, 12.0)


def test_resolve_candle_ohlc_from_metrics_invalid():
    assert _resolve_candle_ohlc({"closed_candle_ohlc": ["bad", 1, 2, 3]}) is None
    assert _resolve_candle_ohlc({"closed_candle_ohlc": [1, 2]}) is None
    assert _resolve_candle_ohlc({}) is None


def test_resolve_candle_ohlc_from_stream():
    candle = Candle(
        symbol="1HZ75V",
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        time=datetime.now(),
        epoch=123456,
    )
    stream = SimpleNamespace(micro_candles={"1HZ75V": [candle, candle]})
    orch = SimpleNamespace(stream=stream)
    assert _resolve_candle_ohlc({}, orch=orch, symbol="1HZ75V") == (100.0, 110.0, 90.0, 105.0)


def test_resolve_candle_ohlc_from_stream_missing():
    stream = SimpleNamespace(micro_candles={})
    orch = SimpleNamespace(stream=stream)
    assert _resolve_candle_ohlc({}, orch=orch, symbol="1HZ75V") is None


def test_should_skip_wick_rejection_disabled_or_force():
    metrics = {"closed_candle_ohlc": [100.0, 120.0, 99.0, 101.0]}
    cfg = {"skip_wick_rejection": False}
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, cfg) is False
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, {"skip_wick_rejection": True}, force=True) is False
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, None) is False


def test_should_skip_wick_rejection_protects_during_recovery_and_flips():
    cfg = {"skip_wick_rejection": True, "material_pending_min": 0.01}
    # Range = 21, upper wick = 120 - 101 = 19 -> ratio >= 0.45
    m1 = {"closed_candle_ohlc": [100.0, 120.0, 99.0, 101.0], "pending_loss_total": 50.0}
    assert should_skip_wick_rejection(m1, TradeDirection.CALL, cfg) is True
    m2 = {"closed_candle_ohlc": [100.0, 120.0, 99.0, 101.0], "loss_clf_flip": True}
    assert should_skip_wick_rejection(m2, TradeDirection.CALL, cfg) is False
    m3 = {"closed_candle_ohlc": [100.0, 120.0, 99.0, 101.0], "anti_trend_lock_flip": True}
    assert should_skip_wick_rejection(m3, TradeDirection.CALL, cfg) is False


def test_should_skip_wick_rejection_missing_ohlc_or_flat():
    cfg = {"skip_wick_rejection": True}
    assert should_skip_wick_rejection({}, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_wick_rejection({"closed_candle_ohlc": [100.0, 100.0, 100.0, 100.0]}, TradeDirection.CALL, cfg)
        is False
    )


def test_should_skip_wick_rejection_call_triggers():
    cfg = {"skip_wick_rejection": True}
    # Open 100, High 120, Low 95, Close 105. Range = 25. Upper wick = 120 - 105 = 15. Ratio = 15/25 = 0.60 >= 0.45
    metrics = {"closed_candle_ohlc": [100.0, 120.0, 95.0, 105.0], "edge": 0.02}
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "wick_rejection_call"
    assert metrics["signal_status"] == "SKIP:wick_rejection_call"


def test_should_skip_wick_rejection_call_super_edge_passes():
    cfg = {"skip_wick_rejection": True}
    metrics = {"closed_candle_ohlc": [100.0, 120.0, 95.0, 105.0], "cal_side_edge": 0.080}
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_wick_rejection_trend_pullback_passes():
    cfg = {"skip_wick_rejection": True}
    metrics = {
        "closed_candle_ohlc": [100.0, 120.0, 95.0, 105.0],
        "edge": 0.025,
        "trend_direction": "CALL",
    }
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_wick_rejection_put_triggers():
    cfg = {"skip_wick_rejection": True}
    # Open 110, High 112, Low 90, Close 102. Range = 22. Lower wick = 102 - 90 = 12. Ratio = 12/22 = 0.545 >= 0.45
    metrics = {"closed_candle_ohlc": [110.0, 112.0, 90.0, 102.0], "edge": 0.02}
    assert should_skip_wick_rejection(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "wick_rejection_put"
    assert metrics["signal_status"] == "SKIP:wick_rejection_put"


def test_should_skip_wick_rejection_balanced_passes():
    cfg = {"skip_wick_rejection": True}
    # Open 100, High 110, Low 90, Close 108. Range = 20. Upper wick = 2. Ratio = 0.10. Lower wick = 10. Ratio = 0.50
    metrics = {"closed_candle_ohlc": [100.0, 110.0, 90.0, 108.0], "edge": 0.02}
    assert should_skip_wick_rejection(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_climactic_blowoff_disabled_or_force():
    metrics = {"closed_candle_ohlc": [100.0, 150.0, 95.0, 148.0], "atr": 10.0}
    cfg = {"skip_climactic_blowoff": False}
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_climactic_blowoff(metrics, TradeDirection.CALL, {"skip_climactic_blowoff": True}, force=True)
        is False
    )
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, None) is False


def test_should_skip_climactic_blowoff_protects_during_recovery_and_flips():
    cfg = {"skip_climactic_blowoff": True, "material_pending_min": 0.01}
    m1 = {"closed_candle_ohlc": [100.0, 150.0, 95.0, 148.0], "atr": 10.0, "pending_loss_total": 20.0}
    assert should_skip_climactic_blowoff(m1, TradeDirection.CALL, cfg) is True
    m2 = {"closed_candle_ohlc": [100.0, 150.0, 95.0, 148.0], "atr": 10.0, "loss_clf_flip": True}
    assert should_skip_climactic_blowoff(m2, TradeDirection.CALL, cfg) is False
    m3 = {"closed_candle_ohlc": [100.0, 150.0, 95.0, 148.0], "atr": 10.0, "anti_trend_lock_flip": True}
    assert should_skip_climactic_blowoff(m3, TradeDirection.CALL, cfg) is False


def test_should_skip_climactic_blowoff_missing_atr_or_normal_range():
    cfg = {"skip_climactic_blowoff": True}
    assert should_skip_climactic_blowoff({}, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_climactic_blowoff({"closed_candle_ohlc": [100.0, 110.0, 95.0, 105.0]}, TradeDirection.CALL, cfg)
        is False
    )
    assert (
        should_skip_climactic_blowoff(
            {"closed_candle_ohlc": [100.0, 110.0, 95.0, 105.0], "atr": 0.0}, TradeDirection.CALL, cfg
        )
        is False
    )
    # Range = 15, ATR = 10 -> ratio = 1.5 <= 2.5
    assert (
        should_skip_climactic_blowoff(
            {"closed_candle_ohlc": [100.0, 110.0, 95.0, 105.0], "atr": 10.0}, TradeDirection.CALL, cfg
        )
        is False
    )


def test_should_skip_climactic_blowoff_call_triggers():
    cfg = {"skip_climactic_blowoff": True}
    # Range = 140 - 100 = 40. ATR = 10. Ratio = 4.0 > 2.5. Bullish (138 > 102).
    metrics = {"closed_candle_ohlc": [102.0, 140.0, 100.0, 138.0], "indicators": {"atr_norm": 10.0}, "edge": 0.03}
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "climactic_blowoff_call"
    assert metrics["signal_status"] == "SKIP:climactic_blowoff_call"


def test_should_skip_climactic_blowoff_call_super_edge_passes():
    cfg = {"skip_climactic_blowoff": True}
    metrics = {
        "closed_candle_ohlc": [102.0, 140.0, 100.0, 138.0],
        "indicators": {"atr_raw": 10.0},
        "cal_side_edge": 0.085,
    }
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_climactic_blowoff_put_triggers():
    cfg = {"skip_climactic_blowoff": True}
    # Range = 140 - 100 = 40. ATR = 10. Ratio = 4.0 > 2.5. Bearish (105 < 138).
    metrics = {"closed_candle_ohlc": [138.0, 140.0, 100.0, 105.0], "atr": 10.0, "edge": 0.02}
    assert should_skip_climactic_blowoff(metrics, TradeDirection.PUT, cfg) is True


def test_should_skip_climactic_blowoff_put_neutral_or_wrong_direction():
    cfg = {"skip_climactic_blowoff": True}
    metrics = {"closed_candle_ohlc": [100.0, 140.0, 100.0, 100.0], "atr": 10.0}
    assert should_skip_climactic_blowoff(metrics, TradeDirection.PUT, cfg) is False
    metrics2 = {"closed_candle_ohlc": [102.0, 140.0, 100.0, 138.0], "atr": 10.0, "edge": 0.02}
    assert should_skip_climactic_blowoff(metrics2, TradeDirection.PUT, cfg) is False


def test_should_skip_climactic_blowoff_opposite_candle_passes():
    cfg = {"skip_climactic_blowoff": True}
    # Bearish candle, but trade is CALL -> not a buying climax
    metrics = {"closed_candle_ohlc": [138.0, 140.0, 100.0, 105.0], "atr": 10.0, "edge": 0.02}
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, cfg) is False
    # Bullish candle, but trade is PUT -> not a selling climax
    metrics2 = {"closed_candle_ohlc": [102.0, 140.0, 100.0, 138.0], "atr": 10.0, "edge": 0.02}
    assert should_skip_climactic_blowoff(metrics2, TradeDirection.PUT, cfg) is False


def test_should_skip_adverse_tick_flow_disabled_or_force():
    metrics = {"flow_features": {"price_velocity": -2.0, "micro_tick_acceleration": -1.0}}
    cfg = {"skip_adverse_tick_flow": False}
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, {"skip_adverse_tick_flow": True}, force=True)
        is False
    )
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, None) is False


def test_should_skip_adverse_tick_flow_protects_during_recovery_and_flips():
    cfg = {"skip_adverse_tick_flow": True, "material_pending_min": 0.01}
    # velocity = -2.0 -> score = -2.0 <= -1.2
    m1 = {"flow_features": {"price_velocity": -2.0}, "pending_loss_total": 10.0}
    assert should_skip_adverse_tick_flow(m1, TradeDirection.CALL, cfg) is True
    m2 = {"flow_features": {"price_velocity": -2.0}, "loss_clf_flip": True}
    assert should_skip_adverse_tick_flow(m2, TradeDirection.CALL, cfg) is False
    m3 = {"flow_features": {"price_velocity": -2.0}, "anti_trend_lock_flip": True}
    assert should_skip_adverse_tick_flow(m3, TradeDirection.CALL, cfg) is False


def test_should_skip_adverse_tick_flow_missing_or_invalid_flow():
    cfg = {"skip_adverse_tick_flow": True}
    assert should_skip_adverse_tick_flow({}, TradeDirection.CALL, cfg) is False
    assert should_skip_adverse_tick_flow({"flow_features": "not_a_dict"}, TradeDirection.CALL, cfg) is False
    # fallback on ValueError
    assert (
        should_skip_adverse_tick_flow(
            {"flow_features": {"price_velocity": "bad", "micro_tick_acceleration": "bad"}}, TradeDirection.CALL, cfg
        )
        is False
    )


def test_should_skip_adverse_tick_flow_call_triggers():
    cfg = {"skip_adverse_tick_flow": True}
    # velocity = -1.0, accel = -0.8 -> score = -1.0 + 0.5 * (-0.8) = -1.4 <= -1.2
    metrics = {"flow_features": {"price_velocity": -1.0, "micro_tick_acceleration": -0.8}, "edge": 0.02}
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "adverse_tick_flow_call"
    assert metrics["signal_status"] == "SKIP:adverse_tick_flow_call"


def test_should_skip_adverse_tick_flow_call_super_edge_passes():
    cfg = {"skip_adverse_tick_flow": True}
    metrics = {"flow_features": {"price_velocity": -1.0, "micro_tick_acceleration": -0.8}, "cal_side_edge": 0.060}
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_adverse_tick_flow_put_triggers():
    cfg = {"skip_adverse_tick_flow": True}
    # velocity = 1.0, accel = 0.8 -> score = 1.4 >= 1.2
    metrics = {"flow_features": {"micro_tick_velocity": 1.0, "price_acceleration": 0.8}, "edge": 0.02}
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "adverse_tick_flow_put"


def test_should_skip_adverse_tick_flow_aligned_passes():
    cfg = {"skip_adverse_tick_flow": True}
    # score is positive (+1.0), CALL is aligned -> passes
    metrics = {"flow_features": {"price_velocity": 1.0, "micro_tick_acceleration": 0.0}, "edge": 0.02}
    assert should_skip_adverse_tick_flow(metrics, TradeDirection.CALL, cfg) is False
    # score is negative (-1.0), PUT is aligned -> passes
    metrics2 = {"flow_features": {"price_velocity": -1.0, "micro_tick_acceleration": 0.0}, "edge": 0.02}
    assert should_skip_adverse_tick_flow(metrics2, TradeDirection.PUT, cfg) is False


def test_should_skip_opposing_marubozu_disabled_or_force():
    metrics = {"closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0]}
    cfg = {"skip_opposing_marubozu": False}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, {"skip_opposing_marubozu": True}, force=True)
        is False
    )
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, None) is False


def test_should_skip_opposing_marubozu_missing_ohlc_or_flat():
    cfg = {"skip_opposing_marubozu": True}
    assert should_skip_opposing_marubozu_flow({}, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_opposing_marubozu_flow(
            {"closed_candle_ohlc": [100.0, 100.0, 100.0, 100.0]}, TradeDirection.CALL, cfg
        )
        is False
    )


def test_should_skip_opposing_marubozu_small_body_passes():
    cfg = {"skip_opposing_marubozu": True}
    # range 20, body 10 -> ratio 0.50 < 0.75
    metrics = {"closed_candle_ohlc": [105.0, 110.0, 90.0, 95.0], "edge": 0.02}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_opposing_marubozu_super_edge_passes():
    cfg = {"skip_opposing_marubozu": True}
    # range 20, body 18 -> ratio 0.90 >= 0.75
    metrics = {"closed_candle_ohlc": [110.0, 110.0, 90.0, 92.0], "cal_side_edge": 0.085}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_opposing_marubozu_call_triggers():
    cfg = {"skip_opposing_marubozu": True}
    # Bearish marubozu: open=110, close=91, high=110, low=90 -> range=20, body=19 (0.95), lower_wick=1 (0.05 < 0.15)
    metrics = {"closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "opposing_bearish_marubozu"
    assert metrics["signal_status"] == "SKIP:opposing_bearish_marubozu"


def test_should_skip_opposing_marubozu_call_passes_if_absorbed_lower_wick():
    cfg = {"skip_opposing_marubozu": True}
    # Bearish with lower wick >= 0.15: open=110, high=110, low=85, close=91 -> range=25, body=19 (0.76), lower_wick = (91-85)/25 = 0.24 >= 0.15
    metrics = {"closed_candle_ohlc": [110.0, 110.0, 85.0, 91.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_opposing_marubozu_put_triggers():
    cfg = {"skip_opposing_marubozu": True}
    # Bullish marubozu: open=91, close=110, low=90, high=110 -> range=20, body=19 (0.95), upper_wick=0 (0.0 < 0.15)
    metrics = {"closed_candle_ohlc": [91.0, 110.0, 90.0, 110.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "opposing_bullish_marubozu"
    assert metrics["signal_status"] == "SKIP:opposing_bullish_marubozu"


def test_should_skip_opposing_marubozu_put_passes_if_absorbed_upper_wick():
    cfg = {"skip_opposing_marubozu": True}
    # Bullish with upper wick >= 0.15: open=91, low=90, close=109, high=115 -> range=25, body=18 (0.72) -> wait, body needs >= 0.75
    # open=90, close=110, low=90, high=115 -> range=25, body=20 (0.80), upper_wick = (115 - 110)/25 = 0.20 >= 0.15
    metrics = {"closed_candle_ohlc": [90.0, 115.0, 90.0, 110.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.PUT, cfg) is False


def test_should_skip_opposing_marubozu_aligned_passes():
    cfg = {"skip_opposing_marubozu": True}
    # Bullish candle with CALL direction -> aligned, passes
    metrics = {"closed_candle_ohlc": [91.0, 110.0, 90.0, 110.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False
    # Bearish candle with PUT direction -> aligned, passes
    metrics2 = {"closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0], "edge": 0.03}
    assert should_skip_opposing_marubozu_flow(metrics2, TradeDirection.PUT, cfg) is False


def test_should_skip_climactic_blowoff_low_atr_floor_scaling():
    cfg = {"skip_climactic_blowoff": True}
    metrics = {
        "closed_candle_ohlc": [100.0, 110.0, 100.0, 109.0],
        "indicators": {"atr_raw": 0.05},
        "edge": 0.02,
    }
    assert should_skip_climactic_blowoff(metrics, TradeDirection.CALL, cfg) is True


def test_should_skip_opposing_marubozu_pullback_with_trend_passes():
    cfg = {"skip_opposing_marubozu": True}
    metrics = {
        "closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0],
        "trend_direction": "CALL",
        "edge": 0.025,
    }
    assert should_skip_opposing_marubozu_flow(metrics, TradeDirection.CALL, cfg) is False
