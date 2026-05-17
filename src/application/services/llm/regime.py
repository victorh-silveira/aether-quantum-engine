"""Classificacao de regime de mercado para prompt."""

from __future__ import annotations

import numpy as np

from src.application.services.llm import (
    IndicatorConfig,
    min_bars_for_indicators,
)
from src.application.services.llm.indicators import (
    _hurst_exponent,
    _shannon_entropy,
)


def classify_regime(m15_closes: list[float], m5_closes: list[float], cfg: IndicatorConfig) -> str:
    """Rotula regime via Hurst (Persistencia) e Entropia (Ruido)."""
    m15 = np.asarray(list(m15_closes), dtype=np.float64)
    m5 = np.asarray(list(m5_closes), dtype=np.float64)
    need = min_bars_for_indicators(cfg)

    if m15.size < need or m5.size < need:
        return "indefinido"

    h15 = _hurst_exponent(m15, cfg.hurst_window)
    h5 = _hurst_exponent(m5, cfg.hurst_window)
    ent5 = _shannon_entropy(m5, cfg.entropy_bins, cfg.entropy_window)

    if ent5 is not None and ent5 > 4.0:
        return "HIGH_ENTROPY_REGIME"

    avg_h = (h15 + h5) / 2.0 if h15 is not None and h5 is not None else (h5 or 0.5)

    if avg_h > 0.55:
        return "trend_persistente"
    if avg_h > 0.51:
        return "trend_fraca"
    if avg_h < 0.49:
        return "mean_reverting"

    return "random_walk"


def sigma_pct_m5(m5_closes: list[float], cfg: IndicatorConfig) -> float | None:
    """Retorna Entropia do M5 como proxy de volatilidade/ruido para o prompt."""
    m5 = np.asarray(list(m5_closes), dtype=np.float64)
    if m5.size < min_bars_for_indicators(cfg):
        return None
    return _shannon_entropy(m5, cfg.entropy_bins, cfg.entropy_window)
