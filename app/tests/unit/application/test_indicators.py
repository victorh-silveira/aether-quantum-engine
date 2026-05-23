import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm.indicators import (
    IndicatorConfig,
    compact_indicators_line,
    min_bars_for_indicators,
    resolve_indicator_config,
)


def test_resolve_indicator_config_defaults():
    c = resolve_indicator_config(None)
    assert c.entropy_bins == 30
    assert c.hurst_window == 30
    assert c.zscore_window == 10


def test_resolve_indicator_config_empty_mapping():
    assert resolve_indicator_config({}) == IndicatorConfig()


def test_resolve_indicator_config_bool_coercion():
    c = resolve_indicator_config(
        {
            "confluence_include_entropy": "yes",
        }
    )
    assert c.confluence_include_entropy is True


def test_min_bars_for_indicators():
    c = resolve_indicator_config({"hurst_window": 50, "entropy_window": 30})
    assert min_bars_for_indicators(c) >= 50


def test_hurst_exponent_persist():
    arr = np.linspace(100.0, 200.0, 100)
    h = ti._hurst_exponent(arr, 50)
    assert h > 0.6


def test_hurst_exponent_random():
    np.random.seed(42)
    arr = 100.0 + np.cumsum(np.random.randn(200))
    h = ti._hurst_exponent(arr, 150)
    assert h is not None
    assert 0.35 < h < 0.65


def test_shannon_entropy_noise():
    arr = np.linspace(100.0, 200.0, 50)
    e = ti._shannon_entropy(arr, 10, 30)
    assert e is not None

    np.random.seed(42)
    arr_noise = np.random.randn(50)
    e_noise = ti._shannon_entropy(arr_noise, 10, 30)
    assert e_noise is not None and e_noise > 2.0


def test_shannon_entropy_single_bin():
    """Cobre a linha 104 (probs.size <= 1) usando bins=1."""
    cfg = IndicatorConfig(entropy_bins=1)
    arr = np.linspace(100.0, 110.0, 50)
    e = ti._shannon_entropy(arr, cfg.entropy_bins, 30)
    assert e == 0.0


def test_z_score_reversion():
    arr = np.ones(50) * 100.0
    arr[-1] = 110.0
    z = ti._z_score_last(arr, 20)
    assert z > 3.0


def test_price_derivatives():
    arr = np.array([100, 101, 103, 106, 110], dtype=float)
    v, a = ti._price_derivatives(arr, 3)
    assert v > 0
    assert a > 0


def test_compact_indicators_line_quant():
    c = list(np.linspace(100.0, 110.0, 120))
    t = compact_indicators_line("M5", c)
    assert "Hurst=" in t
    assert "Entropy=" in t
    assert "Z-Score=" in t
    assert "Sigma=" in t
