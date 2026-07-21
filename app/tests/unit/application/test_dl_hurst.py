import numpy as np

from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio


def test_hurst_returns_neutral_for_short_series():
    prices = np.array([1.0, 1.01, 1.02], dtype=np.float64)
    out = hurst_exponent(prices, window=64, min_window=8)
    assert out.shape == prices.shape
    assert np.allclose(out, 0.5)


def test_variance_ratio_finite():
    prices = np.linspace(100.0, 110.0, 40)
    out = variance_ratio(prices, short=2, long=8)
    assert out.shape == prices.shape
    assert np.isfinite(out).all()
