import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    TRADITIONAL_FEATURE_DIM,
    build_feature_matrix,
    build_feature_row,
    precompute_price_series,
)


def test_feature_dim_is_nineteen():
    assert TRADITIONAL_FEATURE_DIM == 9
    assert FEATURE_DIM == 19


def test_log_return_and_ema_distances():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.0, 105.0], dtype=np.float64)
    series = precompute_price_series(prices, granularity=60, symbol="R_50")
    assert "log_return" in series
    assert "ema_dist_20" in series
    assert "ema_dist_50" in series
    assert "delta_rsi" in series
    assert "roc" in series
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
