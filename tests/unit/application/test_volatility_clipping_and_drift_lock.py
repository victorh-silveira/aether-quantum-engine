import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series


def test_volatility_clipping_in_precompute_price_series():
    prices = np.ones(1024, dtype=np.float64) * 100.0
    prices[-1] = 150.0
    high = prices + 1.0
    low = prices - 1.0

    series = precompute_price_series(
        prices,
        granularity=15,
        symbol="R_10",
        high=high,
        low=low,
    )

    assert np.all(series["bb_width"] >= -3.0)
    assert np.all(series["bb_width"] <= 3.0)
    assert np.all(series["atr_norm"] >= -3.0)
    assert np.all(series["atr_norm"] <= 3.0)
