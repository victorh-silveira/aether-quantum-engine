"""Testes de cobertura para gaps em sniper_payload."""

import numpy as np

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.sniper_payload import entropy_token, hurst_token, zscore_token


def test_sniper_tokens_random_low_high():
    """Cobre os retornos random, low e high nos tokens do Sniper."""
    cfg = IndicatorConfig(hurst_window=15, zscore_window=10, entropy_window=10)
    rng = np.random.default_rng(42)
    rets_random = rng.standard_normal(50) * 0.001
    closes_random = (100.0 * np.exp(np.cumsum(rets_random))).tolist()
    assert hurst_token(closes_random, cfg) == "random"

    closes_low = [100.0] * 19 + [80.0]
    assert zscore_token(closes_low, cfg) == "low"

    covered_high = False
    covered_low = False

    for scale in [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rng_loop = np.random.default_rng(int(scale * 1000) + 42)
        closes = (100.0 + np.arange(30) * 0.1 + rng_loop.uniform(-scale, scale, 30)).tolist()
        res = entropy_token(closes, cfg)
        if res == "high":
            covered_high = True
        elif res == "low":
            covered_low = True

    if not covered_low and entropy_token([100.0] * 30, cfg) == "low":
        covered_low = True

    assert covered_high, "Não conseguiu gerar entropia 'high'"
    assert covered_low, "Não conseguiu gerar entropia 'low'"

    rng_extreme = np.random.default_rng(999)
    closes_extreme = (100.0 + rng_extreme.uniform(-50, 50, 30)).tolist()
    assert entropy_token(closes_extreme, cfg) == "extreme"
