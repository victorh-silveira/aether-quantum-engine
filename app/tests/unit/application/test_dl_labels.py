import numpy as np

from src.application.services.deep_learning.dl_horizon import (
    resolve_implied_vol_bars,
    resolve_label_horizon_bars,
    resolve_label_ma_window,
    resolve_label_mode,
    resolve_label_smooth_bars,
)
from src.application.services.deep_learning.dl_labels import (
    LABEL_MODE_MA_TREND,
    LABEL_MODE_SPOT,
    binary_label_at_index,
    sequence_labels,
)


def test_label_horizon_one_bar_for_60s_contract():
    horizon = resolve_label_horizon_bars(60, {"duration": 60, "duration_unit": "s"}, {})
    assert horizon == 1


def test_label_horizon_one_bar_for_900s_contract_m15():
    horizon = resolve_label_horizon_bars(900, {"duration": 900, "duration_unit": "s"}, {})
    assert horizon == 1


def test_binary_label_rise_spot():
    prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
    assert binary_label_at_index(prices, 0, 1, smooth_bars=1, label_mode=LABEL_MODE_SPOT) is True
    assert binary_label_at_index(prices, 1, 1, smooth_bars=1, label_mode=LABEL_MODE_SPOT) is False


def test_binary_label_ma_trend():
    prices = np.array([100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    assert binary_label_at_index(prices, 2, 1, smooth_bars=3, label_mode=LABEL_MODE_MA_TREND, ma_window=3) is True
    prices_down = np.array([104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0], dtype=np.float64)
    assert binary_label_at_index(prices_down, 2, 1, smooth_bars=2, label_mode=LABEL_MODE_MA_TREND, ma_window=3) is False


def test_resolve_label_helpers():
    assert resolve_label_smooth_bars({}) == 1
    assert resolve_label_smooth_bars({"label_smooth_bars": 5}) == 5
    assert resolve_label_ma_window({"label_ma_window": 8}) == 8
    assert resolve_label_mode({}) == "ma_trend"
    assert resolve_label_mode({"label_mode": "spot_forward"}) == "spot_forward"
    assert resolve_implied_vol_bars({"implied_vol_bars": 60}) == 60


def test_sequence_labels_shape():
    prices = np.linspace(100.0, 120.0, 80)
    targets, masks = sequence_labels(prices, lookback=48, horizon_bars=1, smooth_bars=1)
    assert len(targets) == len(masks) == 80 - 48 - 1
    assert masks.sum() == len(masks)


def test_sequence_labels_shape_with_smooth():
    prices = np.linspace(100.0, 120.0, 80)
    targets, masks = sequence_labels(prices, lookback=48, horizon_bars=1, smooth_bars=5)
    assert len(targets) == len(masks) == 80 - 48 - 5
