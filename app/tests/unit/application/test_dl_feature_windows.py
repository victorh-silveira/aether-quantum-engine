import numpy as np

from src.application.services.deep_learning.dl_feature_build import _feature_windows, precompute_price_series


def test_feature_windows_for_60s_granularity():
    windows = _feature_windows(60)
    assert windows["rsi_period"] == 14
    assert windows["bb_window"] == 20


def test_precompute_price_series_includes_hurst_and_micro():
    prices = np.linspace(100.0, 101.0, 80)
    series = precompute_price_series(prices, granularity=60, symbol="RDBEAR")
    assert "hurst" in series
    assert "bb_pct_b" in series
    assert "tick_count" in series
    assert series["vol_vs_target"].shape == prices.shape


def test_precompute_price_series_uses_open_when_high_low_missing():
    prices = np.array([100.0, 102.0, 101.0, 103.0, 104.0], dtype=np.float64)
    open_ = np.array([99.0, 101.0, 102.0, 102.5, 103.5], dtype=np.float64)
    series = precompute_price_series(prices, open_=open_)
    assert series["atr_norm"].shape == prices.shape
    assert np.isfinite(series["atr_norm"]).all()
