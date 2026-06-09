import numpy as np

from src.application.services.deep_learning.dl_binary_direction import (
    apply_mean_reversion_override,
    binary_direction_veto,
    build_binary_context,
    pair_spread_supports_direction,
    variance_ratio,
)
from src.application.services.deep_learning.dl_params import parse_binary_signal_params
from src.domain.models.trade import TradeDirection


def _trend_prices(start: float, end: float, n: int = 80) -> np.ndarray:
    return np.linspace(start, end, n, dtype=np.float64)


def test_variance_ratio_random_walk_near_one():
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 0.001, 200)
    prices = 100.0 * np.cumprod(1.0 + steps)
    returns = np.zeros(len(prices))
    returns[1:] = np.diff(prices) / (prices[:-1] + 1e-10)
    vr = variance_ratio(returns)
    assert 0.5 < vr < 1.8


def test_variance_ratio_short_series_returns_one():
    assert variance_ratio(np.array([0.0, 0.01, 0.0])) == 1.0


def test_build_binary_context_empty_on_short_series():
    assert build_binary_context(np.array([1.0, 2.0])) == {}


def test_pair_spread_supports_bull_call():
    assert pair_spread_supports_direction(
        TradeDirection.CALL,
        0.5,
        sym_is_bull=True,
        against_limit=1.0,
    )
    assert not pair_spread_supports_direction(
        TradeDirection.CALL,
        -1.5,
        sym_is_bull=True,
        against_limit=1.0,
    )


def test_pair_spread_supports_bear_and_put():
    assert pair_spread_supports_direction(
        TradeDirection.PUT,
        -0.5,
        sym_is_bull=True,
        against_limit=1.0,
    )
    assert pair_spread_supports_direction(
        TradeDirection.CALL,
        -0.5,
        sym_is_bull=False,
        against_limit=1.0,
    )
    assert pair_spread_supports_direction(
        TradeDirection.PUT,
        0.5,
        sym_is_bull=False,
        against_limit=1.0,
    )


def test_mean_reversion_override_at_extreme_z():
    params = {"binary_signal": parse_binary_signal_params({"binary_signal": {"sma_z_extreme": 0.004}})}
    ctx = {"sma_z": 0.01}
    direction, overridden, raw = apply_mean_reversion_override(
        TradeDirection.CALL,
        0.52,
        ctx,
        params,
    )
    assert overridden is True
    assert direction == TradeDirection.PUT
    assert abs(raw - 0.48) < 1e-9


def test_mean_reversion_override_low_extreme_and_strong_dl_skip():
    params = {"binary_signal": parse_binary_signal_params({"binary_signal": {"sma_z_extreme": 0.004}})}
    low_ctx = {"sma_z": -0.01}
    direction, overridden, raw = apply_mean_reversion_override(
        TradeDirection.PUT,
        0.52,
        low_ctx,
        params,
    )
    assert overridden is True
    assert direction == TradeDirection.CALL
    assert abs(raw - 0.48) < 1e-9
    strong_ctx = {"sma_z": 0.01}
    direction2, overridden2, raw2 = apply_mean_reversion_override(
        TradeDirection.CALL,
        0.80,
        strong_ctx,
        params,
    )
    assert overridden2 is False
    assert direction2 == TradeDirection.CALL
    assert raw2 == 0.80
    same_dir, same_override, _ = apply_mean_reversion_override(
        TradeDirection.PUT,
        0.52,
        {"sma_z": 0.01},
        params,
    )
    assert same_override is False
    assert same_dir == TradeDirection.PUT


def test_binary_veto_noise_floor():
    params = {"binary_signal": parse_binary_signal_params({"binary_signal": {"min_rel_vol_execute": 0.9}})}
    ctx = {"rel_vol": 0.1, "sma_z": 0.0, "variance_ratio": 1.0, "has_pair": 0.0}
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "noise_floor"


def test_binary_veto_mean_reversion_call():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.01,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": 0.0,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "mean_reversion"


def test_binary_veto_candle_reject_call():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": -0.002,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
        "close_loc": 0.3,
        "rsi": 0.5,
    }
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "candle_reject"


def test_binary_veto_rsi_exhaust_call():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": 0.002,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
        "close_loc": 0.6,
        "rsi": 0.8,
    }
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "rsi_exhaust"


def test_binary_veto_rsi_exhaust_put():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": -0.002,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
        "close_loc": 0.4,
        "rsi": 0.2,
    }
    assert binary_direction_veto(TradeDirection.PUT, ctx, params, sym_is_bull=True) == "rsi_exhaust"


def test_binary_veto_random_walk_call():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.001,
        "variance_ratio": 0.5,
        "has_pair": 0.0,
        "body": 0.0,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "random_walk"


def test_binary_veto_wick_reject():
    params = {"binary_signal": parse_binary_signal_params({})}
    ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": 0.002,
        "upper_wick": 0.01,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.CALL, ctx, params, sym_is_bull=True) == "wick_reject"


def test_binary_veto_put_paths_and_pair_spread():
    params = {"binary_signal": parse_binary_signal_params({})}
    put_mr = {
        "rel_vol": 1.0,
        "sma_z": -0.01,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": 0.0,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.PUT, put_mr, params, sym_is_bull=True) == "mean_reversion"
    put_rw = {
        "rel_vol": 1.0,
        "sma_z": -0.001,
        "variance_ratio": 0.5,
        "has_pair": 0.0,
        "body": 0.0,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.PUT, put_rw, params, sym_is_bull=True) == "random_walk"
    pair_ctx = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 1.0,
        "z_spread": -2.0,
        "body": 0.0,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
    }
    assert binary_direction_veto(TradeDirection.CALL, pair_ctx, params, sym_is_bull=True) == "pair_spread"
    put_wick = {
        "rel_vol": 1.0,
        "sma_z": 0.0,
        "variance_ratio": 1.0,
        "has_pair": 0.0,
        "body": -0.002,
        "upper_wick": 0.0,
        "lower_wick": 0.01,
    }
    assert binary_direction_veto(TradeDirection.PUT, put_wick, params, sym_is_bull=True) == "wick_reject"
    assert binary_direction_veto(TradeDirection.CALL, {}, params, sym_is_bull=True) is None


def test_build_binary_context_with_ohlc_and_pair():
    n = 40
    prices = _trend_prices(100.0, 110.0, n)
    peer = _trend_prices(200.0, 210.0, n)
    open_ = prices - 0.1
    high = prices + 0.2
    low = prices - 0.3
    ctx = build_binary_context(
        prices,
        pair_prices=peer,
        sym_is_bull=True,
        open_=open_,
        high=high,
        low=low,
    )
    assert ctx["has_pair"] == 1.0
    assert ctx["lower_wick"] > 0.0
    assert "z_spread" in ctx
