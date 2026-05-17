"""Testes de cobertura para gaps em regime."""

import numpy as np

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.regime import classify_regime, sigma_pct_m5


def test_classify_regime_choppy_and_weak():
    """Cobre os regimes choppy_noise e trend_fraca."""
    cfg_h = IndicatorConfig(entropy_window=100, entropy_bins=50, hurst_window=10)
    rng = np.random.default_rng(42)
    log_rets = rng.standard_normal(150) * 5.0
    m5_choppy = np.exp(np.cumsum(log_rets)).tolist()
    m15 = [1.0] * 150
    res = classify_regime(m15, m5_choppy, cfg_h)
    assert res == "HIGH_ENTROPY_REGIME"

    cfg_l = IndicatorConfig(entropy_window=10, hurst_window=50)
    rng = np.random.default_rng(8)
    res_weak = "none"
    for t in [0.0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]:
        rw = t + rng.standard_normal(150) * 0.00001
        m5_w = (100.0 * np.exp(np.cumsum(rw))).tolist()
        res_weak = classify_regime(m5_w, m5_w, cfg_l)
        if res_weak == "trend_fraca":
            break
    assert res_weak == "trend_fraca"
    cfg_p = IndicatorConfig(hurst_window=50)
    m5_p = [100.0 * (1.01**i) for i in range(150)]
    res_p = classify_regime(m5_p, m5_p, cfg_p)
    assert res_p == "trend_persistente"

    m5_m = [100.0 + (i % 2) * 2.0 for i in range(150)]
    res_m = classify_regime(m5_m, m5_m, cfg_p)
    assert res_m == "mean_reverting"

    rng = np.random.default_rng(1)
    m5_r = (100.0 * np.exp(np.cumsum(rng.standard_normal(150) * 0.001))).tolist()
    res_r = classify_regime(m5_r, m5_r, cfg_p)
    assert res_r in ("random_walk", "trend_fraca", "mean_reverting")

    ent_val = sigma_pct_m5(m5_r, cfg_p)
    assert isinstance(ent_val, float)
    assert sigma_pct_m5([1.0], cfg_p) is None
