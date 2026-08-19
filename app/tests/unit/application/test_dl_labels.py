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
    LabelSpec,
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


def test_label_spec_embargo_bars():
    spec = LabelSpec(horizon_bars=1, smooth_bars=5)
    assert spec.embargo_bars == 5
    assert LabelSpec.from_dl_config({"label_horizon_bars": 2, "label_smooth_bars": 3}).embargo_bars == 4


def test_binary_label_supertrend_atr():
    from src.application.services.deep_learning.dl_labels import LABEL_MODE_SUPERTREND_ATR, _supertrend_direction

    prices = np.linspace(100.0, 150.0, 50)
    assert _supertrend_direction(prices, 5) == 1
    assert _supertrend_direction(prices, 40) == 1
    assert binary_label_at_index(prices, 20, 5, label_mode=LABEL_MODE_SUPERTREND_ATR) is True

    prices_down = np.linspace(150.0, 100.0, 50)
    assert _supertrend_direction(prices_down, 40) == -1
    assert binary_label_at_index(prices_down, 20, 5, label_mode=LABEL_MODE_SUPERTREND_ATR) is False
    prices_flat = np.full(50, 100.0)
    assert _supertrend_direction(prices_flat, 40) == 1
    prices_flat_step = np.array([100.0] * 30 + [100.1, 100.0])
    assert _supertrend_direction(prices_flat_step, 31) == -1
