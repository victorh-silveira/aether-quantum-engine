"""Testes unitarios para dl_regime_filter.py com polaridade direcional e confluencia 3D."""

import numpy as np

from src.application.services.deep_learning.dl_regime_filter import (
    is_market_disequilibrium,
    is_price_disequilibrium,
    market_disequilibrium_polarity,
)


def test_market_disequilibrium_polarity_empty():
    assert market_disequilibrium_polarity({}, 0) == 0


def test_market_disequilibrium_polarity_oversold_call():
    series_oversold = {
        "bb_pct_b": np.array([0.03, 0.50]),
        "rsi": np.array([0.18, 0.50]),
        "stoch_k": np.array([0.12, 0.50]),
    }
    assert market_disequilibrium_polarity(series_oversold, 0) == 1
    assert is_market_disequilibrium(series_oversold, 0) is True


def test_market_disequilibrium_polarity_overbought_put():
    series_overbought = {
        "bb_pct_b": np.array([0.97, 0.50]),
        "rsi": np.array([0.82, 0.50]),
        "stoch_k": np.array([0.88, 0.50]),
    }
    assert market_disequilibrium_polarity(series_overbought, 0) == -1
    assert is_market_disequilibrium(series_overbought, 0) is True


def test_market_disequilibrium_polarity_neutral():
    series_neutral = {
        "bb_pct_b": np.array([0.50]),
        "rsi": np.array([0.50]),
        "stoch_k": np.array([0.50]),
    }
    assert market_disequilibrium_polarity(series_neutral, 0) == 0
    assert is_market_disequilibrium(series_neutral, 0) is False


def test_is_market_disequilibrium_confluence_3_vs_2():
    assert is_market_disequilibrium({}, 0) is True
    series_pair = {
        "bb_pct_b": np.array([0.03]),
        "rsi": np.array([0.19]),
    }
    assert is_market_disequilibrium(series_pair, 0) is True
    assert is_market_disequilibrium(series_pair, 0, min_confluence_score=3) is False

    series_trio = {
        "bb_pct_b": np.array([0.03]),
        "rsi": np.array([0.19]),
        "stoch_k": np.array([0.12]),
    }
    assert is_market_disequilibrium(series_trio, 0) is True


def test_market_disequilibrium_polarity_keltner_and_momentum_paths():
    assert market_disequilibrium_polarity({"keltner_pct_b": np.array([0.01]), "rsi": np.array([0.1])}, 0) == 1
    assert market_disequilibrium_polarity({"keltner_pct_b": np.array([0.99]), "rsi": np.array([0.9])}, 0) == -1
    assert market_disequilibrium_polarity({"delta_rsi": np.array([-0.1]), "stoch_k": np.array([0.1])}, 0) == 1
    assert market_disequilibrium_polarity({"delta_rsi": np.array([0.1]), "stoch_k": np.array([0.9])}, 0) == -1
    assert market_disequilibrium_polarity({"ema_9_21_dist": np.array([-0.01]), "rsi": np.array([0.1])}, 0) == 1
    assert market_disequilibrium_polarity({"ema_9_21_dist": np.array([0.01]), "rsi": np.array([0.9])}, 0) == -1
    assert (
        market_disequilibrium_polarity(
            {"macd": np.array([-0.01]), "macd_signal": np.array([0.0]), "rsi": np.array([0.1])}, 0
        )
        == 1
    )
    assert (
        market_disequilibrium_polarity(
            {"macd": np.array([0.01]), "macd_signal": np.array([0.0]), "rsi": np.array([0.9])}, 0
        )
        == -1
    )


def test_is_price_disequilibrium_edges():
    assert is_price_disequilibrium(np.array([100.0]), 0) is True
    assert is_price_disequilibrium(np.array([100.0, 101.0]), 0) is True
    assert is_price_disequilibrium(np.array([100.0, 100.0]), 1) is True
    flat = np.full(20, 100.0)
    assert is_price_disequilibrium(flat, 10) is False
    impulse = np.array([100.0] * 19 + [120.0])
    assert is_price_disequilibrium(impulse, 19) is True
    normal = np.linspace(100.0, 100.1, 20)
    assert is_price_disequilibrium(normal, 19) is False
