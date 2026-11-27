"""Testes unitarios para indicadores tecnicos causais unificados."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.indicators.technical_indicators import (
    IndicatorSnapshot,
    _calc_bands_and_hurst,
    _calc_cci_cmo_wr_roc,
    _calc_macd,
    _calc_rsi,
    _calc_stochastic,
    _ema_series,
    _wilder_adx,
    _wilder_atr,
    compute_indicators,
)
from src.domain.models.market_data import Candle


def _make_candles(count: int = 80, *, step: float = 1.2, base_epoch: int = 1_764_000_000) -> list[Candle]:
    """Gera serie sintetica de velas M5 fechadas."""
    candles: list[Candle] = []
    price = 6500.0
    for idx in range(count):
        delta = step if idx % 5 != 3 else -step * 0.4
        op = price
        cl = price + delta
        hi = max(op, cl) + 0.5
        lo = min(op, cl) - 0.5
        ep = base_epoch + idx * 300
        candles.append(Candle("R_10", op, hi, lo, cl, datetime.fromtimestamp(ep, tz=UTC), ep))
        price = cl
    return candles


def test_compute_indicators_full_snapshot():
    candles = _make_candles(80, step=1.5)
    snap = compute_indicators(candles)
    assert isinstance(snap, IndicatorSnapshot)
    as_dict = snap.to_dict()
    assert snap.close == candles[-1].close
    assert snap.ema_9 > snap.ema_50
    assert snap.rsi_14 > 50.0
    assert snap.adx_14 > 0.0
    assert snap.atr_14 > 0.0
    assert 0.0 <= snap.hurst_14 <= 1.0
    assert "session_zscore" in as_dict
    assert as_dict["candle_dir"] in (-1.0, 0.0, 1.0)


def test_compute_indicators_edge_and_short_series():
    with pytest.raises(ValueError, match="vazia"):
        compute_indicators([])

    short = _make_candles(3, step=0.0)
    snap = compute_indicators(short)
    assert snap.macd == 0.0
    assert snap.rsi_14 == 50.0
    assert snap.stoch_k == 50.0
    assert snap.adx_14 == 0.0
    assert snap.bollinger_pct_b == 0.5

    # Serie flat e extremos
    assert _ema_series([], 9) == []
    assert _wilder_atr(short[:1], 14) == 0.0
    assert _wilder_atr(short, 0) == 0.0
    assert _wilder_adx(short, 14) == (0.0, 0.0, 0.0, 0.0)
    assert _calc_rsi([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], 7) == 100.0
    wilder_rsi = _calc_rsi([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 13.5, 14.0, 13.2], 7)
    assert 50.0 < wilder_rsi < 100.0
    assert _calc_rsi([10.0] * 10, 7) == 50.0
    assert _calc_macd([10.0] * 10) == (0.0, 0.0, 0.0, 0.0)
    assert _calc_stochastic(short, 14, 3) == (50.0, 50.0)
    assert _calc_cci_cmo_wr_roc(short, [10.0, 10.0, 10.0], 14) == (0.0, 0.0, -50.0, 0.0)
    assert _calc_bands_and_hurst([10.0] * 5, 10.0, 0.0) == (0.5, 0.0, 0.5, 0.5)
    _, _, _, hurst_50 = _calc_bands_and_hurst([100.0 + i * 0.5 for i in range(60)], 125.0, 1.0)
    assert 0.0 <= hurst_50 <= 1.0
    flat_bars = [
        Candle(
            "R_10",
            10.0,
            10.0,
            10.0,
            10.0,
            datetime.fromtimestamp(1_764_000_000 + i * 300, tz=UTC),
            1_764_000_000 + i * 300,
        )
        for i in range(25)
    ]
    assert _wilder_adx(flat_bars, 14) == (0.0, 0.0, 0.0, 0.0)
    assert compute_indicators(flat_bars).streak == 0
