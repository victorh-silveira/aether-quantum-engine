import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    TRADITIONAL_FEATURE_DIM,
    build_feature_matrix,
    build_feature_row,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_feature_indicators import (
    calculate_adx,
    calculate_ema_crossover,
    calculate_stochastic,
    calculate_volatility_ratio,
    calculate_williams_r,
)


def test_feature_dim_is_thirty_two():
    assert TRADITIONAL_FEATURE_DIM == 20
    assert FEATURE_DIM == 32


def test_log_return_and_ema_distances():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.0, 105.0], dtype=np.float64)
    series = precompute_price_series(prices, granularity=60, symbol="R_50")
    assert "log_return" in series
    assert "ema_dist_20" in series
    assert "ema_dist_50" in series
    assert "delta_rsi" in series
    assert "roc" in series
    assert "price_zscore" in series
    assert "implied_vol_ratio" in series
    assert "macd" in series
    assert "stoch_k" in series
    assert "cci" in series
    assert "adx" in series
    assert "di_diff" in series
    assert "williams_r" in series
    assert "ema_9_21_dist" in series
    assert "roc_rsi" in series
    assert "vol_ratio_short_long" in series
    assert np.isclose(series["log_return"][1], np.log(101.0 / 100.0), rtol=1e-5)
    assert np.isfinite(series["ema_dist_50"]).all()
    assert np.isfinite(series["roc"]).all()


def test_build_feature_row_shape():
    prices = np.linspace(100.0, 110.0, 80)
    series = precompute_price_series(prices, granularity=60, symbol="R_75")
    row = build_feature_row(series, 40)
    assert row.shape == (FEATURE_DIM,)
    matrix = build_feature_matrix(series)
    assert matrix.shape == (len(prices), FEATURE_DIM)


def test_calculate_stochastic_flat_prices():
    prices = np.full(50, 100.0)
    k, d = calculate_stochastic(prices, prices, prices, period=14)
    assert (k == 0.5).all()


def test_new_indicators_edge_cases():
    # Empty / too short arrays
    prices = np.array([100.0, 100.0])
    adx, di_diff = calculate_adx(prices, prices, prices, period=14)
    assert (adx == 0.0).all()
    assert (di_diff == 0.0).all()

    wr = calculate_williams_r(prices, prices, prices, period=14)
    assert (wr == 0.5).all()

    vr = calculate_volatility_ratio(prices, short=5, long=20)
    assert (vr == 0.0).all()

    # Flat prices
    prices = np.full(50, 100.0)
    adx, di_diff = calculate_adx(prices, prices, prices, period=14)
    assert np.isfinite(adx).all()
    assert np.isfinite(di_diff).all()

    wr = calculate_williams_r(prices, prices, prices, period=14)
    assert (wr == 0.5).all()

    # Vol ratio flat prices / zero std
    log_ret = np.zeros(50)
    vr = calculate_volatility_ratio(log_ret, short=5, long=20)
    assert (vr[:19] == 0.0).all()
    assert (vr[19:] == 1.0).all()

    # EMA crossover
    dist = calculate_ema_crossover(prices, fast=9, slow=21)
    assert (dist == 0.0).all()
