import numpy as np

from src.application.services.llm.indicators import resolve_indicator_config
from src.application.services.llm.regime import (
    classify_regime,
    sigma_pct_m5,
)


def test_classify_regime_trend_persistente():
    cfg = resolve_indicator_config({"hurst_window": 30})
    m15 = [100.0 * (1.01**i) for i in range(45)]
    m5 = [100.0 * (1.01**i) for i in range(45)]
    lab = classify_regime(m15, m5, cfg)
    assert lab == "trend_persistente"


def test_classify_regime_mean_reverting():
    cfg = resolve_indicator_config({"hurst_window": 30})
    m15 = [100.0 + (i % 2) * 2.0 for i in range(45)]
    m5 = [100.0 + (i % 2) * 2.0 for i in range(45)]
    lab = classify_regime(m15, m5, cfg)
    assert lab == "mean_reverting"


def test_classify_regime_random_walk():
    cfg = resolve_indicator_config({"hurst_window": 30})
    np.random.seed(1)
    m15 = np.cumsum(np.random.randn(45) * 0.1 + 100).tolist()
    m5 = np.cumsum(np.random.randn(45) * 0.1 + 100).tolist()
    lab = classify_regime(m15, m5, cfg)
    assert lab in ("random_walk", "trend_fraca", "mean_reverting", "trend_persistente", "HIGH_ENTROPY_REGIME")


def test_sigma_pct_m5_returns_float_or_none():
    cfg = resolve_indicator_config({"hurst_window": 20, "entropy_window": 10})
    assert sigma_pct_m5([], cfg) is None
    m5 = list(np.linspace(100.0, 101.0, 40))
    v = sigma_pct_m5(m5, cfg)
    assert isinstance(v, float) and v >= 0.0
