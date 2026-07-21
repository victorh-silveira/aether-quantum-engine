import numpy as np

from src.application.services.deep_learning.dl_congestion import series_last, squeeze_congestion_active
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


def test_series_last_defaults():
    assert series_last({}, "adx", 1.0) == 1.0
    assert series_last({"adx": [0.2, 0.3]}, "adx") == 0.3


def test_squeeze_congestion_active_uses_settings_thresholds():
    cfg = load_indicator_config_from_settings()
    congestion = cfg["congestion"]
    prices = np.full(int(congestion["min_bars"]) + 5, 100.0)
    series = {"adx": [float(congestion["adx_max"]) - 0.01]}
    assert (
        squeeze_congestion_active(
            prices,
            series,
            bb_window=int(cfg["windows"]["bb_window"]),
            bb_std_mult=float(cfg["multipliers"]["bb_std_mult"]),
            congestion=congestion,
        )
        is True
    )
