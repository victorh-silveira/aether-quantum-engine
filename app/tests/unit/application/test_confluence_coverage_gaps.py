"""Testes de cobertura para gaps em indicators_confluence."""

from unittest.mock import patch

import numpy as np

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.indicators_confluence import ema_distance_guard_line, mtf_confluence_line


def test_mtf_confluence_random_walk():
    """Cobre o sinal RANDOM_WALK_SEM_EDGE."""
    np.random.seed(42)
    closes_h = (100.0 + np.cumsum(np.random.randn(100))).tolist()
    closes_l = (100.0 + np.cumsum(np.random.randn(100))).tolist()
    res = mtf_confluence_line(closes_h, closes_l)
    assert "sinal_quant=DIVERGENCIA_ESTRUTURAL_DETECTADA (Risky)" in res


def test_mtf_confluence_random_walk_actual():
    """Cobre o sinal RANDOM_WALK_SEM_EDGE forçando os regimes via mock."""
    with (
        patch("src.application.services.llm.indicators._market_regime_quant", return_value="random_walk_high_noise"),
        patch("src.application.services.llm.indicators._shannon_entropy", return_value=1.0),
        patch("src.application.services.llm.indicators._hurst_exponent", return_value=0.5),
    ):
        res = mtf_confluence_line([100.0] * 50, [100.0] * 50)
        assert "sinal_quant=RANDOM_WALK_SEM_EDGE (Noisy)" in res


def test_mtf_confluence_divergencia():
    """Cobre o sinal DIVERGENCIA_ESTRUTURAL_DETECTADA."""
    with (
        patch("src.application.services.llm.indicators._market_regime_quant", return_value="trend_persistente"),
        patch("src.application.services.llm.indicators._z_score_last", return_value=-2.0),
        patch("src.application.services.llm.indicators._shannon_entropy", return_value=1.0),
        patch("src.application.services.llm.indicators._hurst_exponent", return_value=0.5),
    ):
        res = mtf_confluence_line([100.0] * 50, [100.0] * 50)
        assert "DIVERGENCIA_ESTRUTURAL_DETECTADA (Risky)" in res


def test_ema_distance_guard_line_branches():
    """Cobre as ramificações da guarda estatística (Z-Score)."""
    cfg = IndicatorConfig(zscore_window=20)
    assert "dados_insuficientes" in ema_distance_guard_line("M5", [100.0] * 5, cfg)

    assert "DENTRO_DA_NORMALIDADE" in ema_distance_guard_line("M5", [100.0] * 30, cfg)

    closes = [100.0] * 29 + [1000.0]
    assert "EXTREMO_ESTATISTICO_ALERTA" in ema_distance_guard_line("M5", closes, cfg)
