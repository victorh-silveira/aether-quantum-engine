import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_pair_features import (
    align_pair_lengths,
    pair_feature_row,
    precompute_pair_series,
    spread_confirms_direction,
)


def test_align_pair_lengths_empty():
    a, b = align_pair_lengths(np.array([]), np.array([1.0]))
    assert a.size == 0 and b.size == 0


def test_precompute_pair_series_short():
    out = precompute_pair_series(np.array([1.0]), np.array([2.0]))
    assert out["spread"].shape == (1,)


def test_precompute_pair_series_rolling():
    bull = np.linspace(100.0, 110.0, 40)
    bear = np.linspace(50.0, 48.0, 40)
    out = precompute_pair_series(bull, bear)
    assert out["z_spread"][-1] != 0.0
    assert abs(out["corr"][-1]) <= 1.0
    row = pair_feature_row(out, 25)
    assert row.shape == (3,)


def test_spread_confirms_direction_bull_up():
    bull = np.array([10.0, 11.0, 12.0])
    bear = np.array([5.0, 5.0, 4.9])
    assert spread_confirms_direction(bull, bear, 0, target_up=True, sym_is_bull=True)
    assert not spread_confirms_direction(bull, bear, 0, target_up=False, sym_is_bull=True)


def test_spread_confirms_direction_bear_up():
    bear = np.array([5.0, 5.2, 5.1])
    bull = np.array([10.0, 10.0, 10.1])
    assert spread_confirms_direction(bear, bull, 0, target_up=True, sym_is_bull=False)


def test_spread_confirms_short_index():
    assert spread_confirms_direction(np.array([1.0]), np.array([1.0]), 0, target_up=True, sym_is_bull=True)


def test_spread_confirms_bear_index_guard():
    bear = np.array([5.0])
    bull = np.array([10.0, 10.1])
    assert spread_confirms_direction(bear, bull, 0, target_up=True, sym_is_bull=False)


def test_precompute_nan_inf_robustness():
    bull = np.array([100.0, np.nan, 105.0, np.inf, 110.0, -np.inf, 115.0] * 10, dtype=np.float64)
    bear = np.array([50.0, 51.0, np.nan, -np.inf, 49.0, np.inf, 48.0] * 10, dtype=np.float64)
    out_pair = precompute_pair_series(bull, bear)
    for k, v in out_pair.items():
        assert not np.isnan(v).any(), f"NaN found in key {k}"
        assert not np.isinf(v).any(), f"Inf found in key {k}"


def test_precompute_price_nan_inf_robustness():
    prices = np.array([100.0, np.nan, 105.0, np.inf, 110.0, -np.inf, 115.0] * 10, dtype=np.float64)
    open_ = np.array([100.0, 101.0, np.nan, 102.0, np.inf, 103.0, 104.0] * 10, dtype=np.float64)
    high = np.array([101.0, 102.0, 106.0, np.inf, 111.0, 105.0, 116.0] * 10, dtype=np.float64)
    low = np.array([99.0, np.nan, 104.0, 98.0, 109.0, -np.inf, 114.0] * 10, dtype=np.float64)

    out_price = precompute_price_series(prices, open_=open_, high=high, low=low)
    for k, v in out_price.items():
        assert not np.isnan(v).any(), f"NaN found in key {k}"
        assert not np.isinf(v).any(), f"Inf found in key {k}"
