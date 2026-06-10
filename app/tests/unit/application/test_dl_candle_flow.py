import numpy as np

from src.application.services.deep_learning.dl_binary_direction import build_binary_context
from src.application.services.deep_learning.dl_candle_flow import (
    apply_candle_flow_override,
    augment_binary_context,
    flow_alignment_bonus,
    flow_aligns_with,
    flow_implied_direction,
    flow_strength,
    sma_extreme_direction,
)
from src.domain.models.trade import TradeDirection


def _ohlc_trend(n: int = 80, step: float = 0.003) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prices = 100.0 * np.cumprod(1.0 + np.full(n, step, dtype=np.float64))
    open_ = np.roll(prices, 1)
    open_[0] = prices[0]
    high = prices * 1.001
    low = prices * 0.999
    return prices, open_, high, low


def test_build_binary_context_includes_flow_fields():
    prices, open_, high, low = _ohlc_trend()
    ctx = build_binary_context(prices, open_=open_, high=high, low=low)
    assert "body_sum_3" in ctx
    assert "ema_spread" in ctx
    assert "ret_3" in ctx
    assert "market_trend" in ctx


def test_flow_implied_direction_bullish_trend():
    prices, open_, high, low = _ohlc_trend(step=0.004)
    ctx = build_binary_context(prices, open_=open_, high=high, low=low)
    assert flow_implied_direction(ctx) == TradeDirection.CALL
    assert flow_strength(ctx) > 0.2


def test_flow_implied_direction_bearish_trend():
    prices, open_, high, low = _ohlc_trend(step=-0.004)
    ctx = build_binary_context(prices, open_=open_, high=high, low=low)
    assert flow_implied_direction(ctx) == TradeDirection.PUT


def test_apply_candle_flow_override_flips_weak_dl():
    ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "ret_3": 0.002,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "sma_z": 0.0,
        "rsi": 0.55,
        "rsi_slope": 0.02,
    }
    direction, overridden, raw = apply_candle_flow_override(
        TradeDirection.PUT,
        0.51,
        ctx,
        {"binary_signal": {"weak_dl_override_margin": 0.08}},
    )
    assert overridden is True
    assert direction == TradeDirection.CALL
    assert raw == 0.51


def test_flow_implied_direction_mean_reversion_regime():
    ctx = {
        "body": 0.0,
        "body_sum_3": 0.0,
        "ema_spread": 0.0,
        "ret_5": 0.0,
        "ret_3": 0.0,
        "close_loc": 0.5,
        "variance_ratio": 0.75,
        "sma_z": -0.004,
        "body_streak": 0.0,
        "rsi_slope": 0.0,
    }
    assert flow_implied_direction(ctx) == TradeDirection.CALL


def test_flow_implied_direction_tie_break_close_loc():
    ctx = {
        "body": 0.0,
        "body_sum_3": 0.0,
        "ema_spread": 0.0,
        "ret_5": 0.0,
        "ret_3": 0.0,
        "close_loc": 0.53,
        "variance_ratio": 0.85,
        "sma_z": 0.0,
        "body_streak": 0.0,
        "rsi_slope": 0.0,
    }
    assert flow_implied_direction(ctx) == TradeDirection.CALL


def test_flow_implied_direction_tie_break_put_close_loc():
    ctx = {
        "body": 0.0,
        "body_sum_3": 0.0,
        "ema_spread": 0.0,
        "ret_5": 0.0,
        "ret_3": 0.0,
        "close_loc": 0.47,
        "variance_ratio": 0.85,
        "sma_z": 0.0,
        "body_streak": 0.0,
        "rsi_slope": 0.0,
    }
    assert flow_implied_direction(ctx) == TradeDirection.PUT


def test_flow_implied_direction_empty_context():
    assert flow_implied_direction({}) is None


def test_flow_implied_direction_mid_regime_ema():
    assert (
        flow_implied_direction(
            {
                "body": 0.0,
                "body_sum_3": 0.0,
                "ema_spread": 0.0008,
                "ret_5": 0.0,
                "ret_3": 0.0,
                "close_loc": 0.50,
                "variance_ratio": 0.85,
                "sma_z": 0.0,
                "body_streak": 0.0,
                "rsi_slope": 0.0,
            }
        )
        == TradeDirection.CALL
    )
    assert (
        flow_implied_direction(
            {
                "body": 0.0,
                "body_sum_3": 0.0,
                "ema_spread": -0.0008,
                "ret_5": 0.0,
                "ret_3": 0.0,
                "close_loc": 0.50,
                "variance_ratio": 0.85,
                "sma_z": 0.0,
                "body_streak": 0.0,
                "rsi_slope": 0.0,
            }
        )
        == TradeDirection.PUT
    )


def test_augment_binary_context_body_streak_breaks_on_reversal():
    n = 12
    bodies = np.array([0.001, 0.001, 0.001, -0.001, -0.001], dtype=np.float64)
    series = {
        "body": bodies,
        "ema_spread": np.zeros(n),
        "ret_5": np.zeros(n),
        "rsi_slope": np.zeros(n),
    }
    ctx = {"variance_ratio": 1.0}
    assert augment_binary_context(dict(ctx), series, 4)["body_streak"] == 2.0
    assert augment_binary_context(dict(ctx), series, 3)["body_streak"] == 1.0


def test_apply_candle_flow_override_keeps_direction_when_flow_weak():
    direction, overridden, raw = apply_candle_flow_override(
        TradeDirection.PUT,
        0.51,
        {"body": 0.0001, "close_loc": 0.55},
        {"binary_signal": {"weak_dl_override_margin": 0.08}},
    )
    assert overridden is False
    assert direction == TradeDirection.PUT
    assert raw == 0.51


def test_flow_aligns_with_branches():
    weak_ctx = {"body": 0.0001, "close_loc": 0.55}
    assert flow_aligns_with(TradeDirection.PUT, weak_ctx) is True
    assert flow_aligns_with(TradeDirection.CALL, {}) is True
    strong_ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "rsi_slope": 0.02,
    }
    assert flow_aligns_with(TradeDirection.CALL, strong_ctx) is True
    assert flow_aligns_with(TradeDirection.PUT, strong_ctx) is False


def test_flow_alignment_bonus_misaligned():
    ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "rsi_slope": 0.02,
    }
    assert flow_alignment_bonus(TradeDirection.PUT, ctx) < 0.0


def test_sma_extreme_direction_branches():
    assert sma_extreme_direction(0.005) == TradeDirection.PUT
    assert sma_extreme_direction(-0.005) == TradeDirection.CALL
    assert sma_extreme_direction(0.0) is None
