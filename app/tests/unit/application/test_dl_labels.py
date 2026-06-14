import numpy as np

from src.application.services.deep_learning.dl_horizon import resolve_label_horizon_bars
from src.application.services.deep_learning.dl_labels import binary_label_at_index, sequence_labels


def test_label_horizon_one_bar_for_60s_contract():
    horizon = resolve_label_horizon_bars(60, {"duration": 60, "duration_unit": "s"}, {})
    assert horizon == 1


def test_binary_label_rise():
    prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
    assert binary_label_at_index(prices, 0, 1) is True
    assert binary_label_at_index(prices, 1, 1) is False


def test_sequence_labels_shape():
    prices = np.linspace(100.0, 120.0, 80)
    targets, masks = sequence_labels(prices, lookback=48, horizon_bars=1)
    assert len(targets) == len(masks) == 80 - 48 - 1
    assert masks.sum() == len(masks)
