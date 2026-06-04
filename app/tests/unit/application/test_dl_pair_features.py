import numpy as np

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
