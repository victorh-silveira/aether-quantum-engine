import numpy as np

from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio


def test_hurst_returns_neutral_for_short_series():
    prices = np.array([1.0, 1.01, 1.02], dtype=np.float64)
    out = hurst_exponent(prices, window=64, min_window=8)
    assert out.shape == prices.shape
    assert np.allclose(out, 0.5)


def test_msd_hurst_is_causal_and_brownian_near_half():
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, size=3000)))
    base = hurst_exponent(prices, window=100, min_window=32)
    modified = prices.copy()
    modified[2000:] *= 1.2
    changed = hurst_exponent(modified, window=100, min_window=32)
    assert np.array_equal(base[:2000], changed[:2000])
    assert 0.35 < float(np.median(base[100:])) < 0.65
    assert np.isfinite(base).all()


def test_msd_hurst_returns_neutral_when_too_few_scales():
    prices = np.linspace(1.0, 2.0, 40)
    assert np.allclose(hurst_exponent(prices, window=4, min_window=4), 0.5)


def test_variance_ratio_finite():
    prices = np.linspace(100.0, 110.0, 40)
    out = variance_ratio(prices, short=2, long=8)
    assert out.shape == prices.shape
    assert np.isfinite(out).all()
