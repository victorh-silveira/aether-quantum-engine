"""Contratos de causalidade e alinhamento entre treino e inferencia."""

import numpy as np
import pytest

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_indicators import calculate_rsi
from src.application.services.deep_learning.dl_sequence_extract import sequence_price_deltas


def test_indicadores_nao_mudam_quando_futuro_e_adicionado():
    prices = 100 + np.sin(np.arange(90) / 3) + np.arange(90) * 0.1
    full = precompute_price_series(prices, high=prices + 1, low=prices - 1)
    for end in (8, 20, 60):
        prefix = precompute_price_series(prices[:end], high=prices[:end] + 1, low=prices[:end] - 1)
        for key in full:
            np.testing.assert_allclose(prefix[key], full[key][:end], atol=1e-7, err_msg=key)


def test_rsi_constante_e_neutro_com_dtype_inteiro():
    np.testing.assert_array_equal(calculate_rsi(np.ones(30, dtype=int), 14), np.full(30, 50.0))


def test_retorno_auxiliar_usa_barra_da_decisao_e_futuro():
    prices = np.array([90.0, 95.0, 100.0, 120.0, 80.0, 110.0])
    result = sequence_price_deltas(prices, 2, label_mode="spot_forward")
    assert result[0] == pytest.approx(0.2)
    smooth = sequence_price_deltas(prices, 2, label_smooth_bars=2, label_mode="spot_forward")
    assert smooth[0] == pytest.approx(0.0)
