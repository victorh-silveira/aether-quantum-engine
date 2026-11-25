"""Equivalencia numerica dos lotes e das janelas escalares de producao."""

import numpy as np
import pytest

from scripts.analysis.frequency_features import batch_features, rolling_di, rolling_ema
from src.indicator.domain import Candle
from src.indicator.pullback_indicators import ema_direct
from src.indicator.strategy_persistent_trend import _resolve_h1_trend, calc_wilder_di_adx


def series(count, period=900):
    """Gera trajetoria fixa com range valido e direcao variavel."""
    rng = np.random.default_rng(17)
    values = 100 + np.cumsum(rng.normal(0, 0.4, count + 1))
    return [
        Candle(i * period, values[i], max(values[i : i + 2]) + 0.2, min(values[i : i + 2]) - 0.2, values[i + 1])
        for i in range(count)
    ]


@pytest.mark.parametrize("period", [14, 20, 24, 28])
def test_exact_rolling_seed(period):
    """Inclui janelas crescentes e deslocamento alem de 500 velas."""
    bars = series(550)
    diff, adx = rolling_di(bars, period)
    for i in (0, period - 1, period, 50, 127, 498, 499, 500, 549):
        expected = calc_wilder_di_adx(bars[max(0, i - 499) : i + 1], period)
        assert (diff[i], adx[i]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert rolling_di([], period)[0].size == 0


def test_ema_seed_and_causal_h1():
    """H1 ainda em formacao nunca influencia a tendencia utilizada em M15."""
    bars = series(550)
    h1 = series(160, 3600)
    values = np.array([b.close for b in h1])
    actual = rolling_ema(values, 20)
    assert actual[-1] == pytest.approx(ema_direct(values.tolist(), 20)[-1])
    features = batch_features(bars, h1)
    for i in (50, 200, 499, 549):
        cutoff = bars[i].epoch + 900
        closed = [b for b in h1 if b.epoch + 3600 <= cutoff][-500:]
        assert features["h1_trend"][i] == _resolve_h1_trend(bars[max(0, i - 499) : i + 1], closed)
    fallback = batch_features(bars, [])["h1_trend"]
    assert fallback[-1] == _resolve_h1_trend(bars[-500:], None)


def test_flat_range():
    """Mercado sem oscilacao permanece sem direcionalidade."""
    bars = [Candle(i * 900, 10, 10, 10, 10) for i in range(60)]
    diff, adx = rolling_di(bars, 20)
    assert not np.any(diff)
    assert not np.any(adx)
