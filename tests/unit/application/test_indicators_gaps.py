"""Testes de cobertura para gaps em indicators."""

import numpy as np

from src.application.services.llm.indicators import (
    _bool_cfg,
    _clamp_float,
    _clamp_int,
    _market_regime_quant,
    _shannon_entropy,
    _vol_range_pct,
)


def test_bool_cfg_branches():
    """Cobre ramos de conversão booleana."""
    assert _bool_cfg(v=True, default=False) is True
    assert _bool_cfg(v=1, default=False) is True
    assert _bool_cfg(v="yes", default=False) is True
    assert _bool_cfg(v={}, default=True) is True


def test_clamp_err_fallback():
    """Cobre fallbacks de erro no clamp."""
    assert _clamp_int("invalido", 0, 10, 5) == 5
    assert _clamp_float("invalido", 0.0, 1.0, 0.5) == 0.5
    assert _clamp_float(2.0, 0.0, 1.0, 0.5) == 1.0


def test_shannon_entropy_low_probs():
    """Cobre caso de probs.size <= 1 na entropia."""
    closes = np.array([100.0] * 50)
    assert _shannon_entropy(closes) == 0.0


def test_vol_range_pct_zero_division():
    """Cobre divisão por zero na volatilidade e amostra curta."""
    closes = np.array([0.0] * 20)
    assert _vol_range_pct(closes, 14) == 0.0
    assert _vol_range_pct(np.array([100.0]), 14) is None


def test_market_regime_quant_random():
    """Cobre o retorno random_walk."""
    assert _market_regime_quant(0.5, 0.0) == "random_walk_high_noise"
