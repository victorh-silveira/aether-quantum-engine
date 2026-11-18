"""Testes unitarios do detector de regime adaptativo de tendencia (R4/R6)."""

from __future__ import annotations

from datetime import datetime

from src.indicator.domain import Candle
from src.indicator.institutional import BRT_TIMEZONE
from src.indicator.pullback_senior import evaluate_senior_institutional
from src.indicator.pullback_trend_regime import evaluate_trend_regime


def _trend_bars(count: int = 10, *, bull: bool = True) -> list[Candle]:
    delta = 1.0 if bull else -1.0
    base = 100.0
    out: list[Candle] = []
    for i in range(count):
        o = base + i * delta
        c = o + delta * 0.6
        hi = max(o, c) + 0.2
        lo = min(o, c) - 0.2
        out.append(Candle(i * 900, o, hi, lo, c))
    return out


def test_evaluate_trend_regime_r4_and_r6_paths():
    bull = _trend_bars(10, bull=True)
    bear = _trend_bars(10, bull=False)
    fast_b = [bull[-1].close - 0.2] * 10
    slow_b = [bull[-1].close - 1.5] * 10
    fast_s = [bear[-1].close + 0.2] * 10
    slow_s = [bear[-1].close + 1.5] * 10

    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=35.0, pdi=30.0, mdi=10.0, m_hist=0.5, hurst=0.62, mode="off"
        )
        is None
    )
    assert (
        evaluate_trend_regime(
            bull[:4], fast_b, slow_b, 1.0, adx=35.0, pdi=30.0, mdi=10.0, m_hist=0.5, hurst=0.62, mode="r4"
        )
        is None
    )
    assert (
        evaluate_trend_regime(
            bull, [bull[-1].close - 2.0], slow_b, 1.0, adx=35.0, pdi=30.0, mdi=10.0, m_hist=0.5, hurst=0.62, mode="r4"
        )
        is None
    )
    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=25.0, pdi=30.0, mdi=10.0, m_hist=0.5, hurst=0.65, mode="r4"
        )
        is None
    )

    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=36.0, pdi=30.0, mdi=10.0, m_hist=0.5, hurst=0.63, mode="r4"
        )
        == "CALL"
    )
    assert (
        evaluate_trend_regime(
            bear, fast_s, slow_s, 1.0, adx=36.0, pdi=10.0, mdi=30.0, m_hist=-0.5, hurst=0.63, mode="r4"
        )
        == "PUT"
    )
    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=36.0, pdi=10.0, mdi=30.0, m_hist=0.5, hurst=0.63, mode="r4"
        )
        is None
    )

    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=32.0, pdi=25.0, mdi=10.0, m_hist=0.5, hurst=0.61, mode="r6"
        )
        == "CALL"
    )
    assert (
        evaluate_trend_regime(
            bear, fast_s, slow_s, 1.0, adx=32.0, pdi=10.0, mdi=25.0, m_hist=-0.5, hurst=0.61, mode="r6"
        )
        == "PUT"
    )
    assert (
        evaluate_trend_regime(
            bull, fast_b, slow_b, 1.0, adx=32.0, pdi=25.0, mdi=10.0, m_hist=-0.5, hurst=0.61, mode="r6"
        )
        is None
    )


def test_senior_institutional_uses_adaptive_trend_fallback(monkeypatch):
    period = 900
    bars = [Candle(i * period, 100.0 + i * 0.5, 100.8 + i * 0.5, 99.8 + i * 0.5, 100.4 + i * 0.5) for i in range(64)]
    closes = [b.close for b in bars]
    fast = [c - 0.1 for c in closes]
    slow = [c - 1.5 for c in closes]
    hour = datetime.fromtimestamp(bars[-1].epoch + period, tz=BRT_TIMEZONE).hour

    monkeypatch.setattr(
        "src.indicator.pullback_senior.session_price_reference",
        lambda b: (closes[-1], closes[-1] + 5.0, closes[-1] - 5.0),
    )
    monkeypatch.setattr("src.indicator.pullback_senior.detect_fvg_mitigation", lambda b, a: (False, False))
    monkeypatch.setattr("src.indicator.pullback_senior.calc_adx", lambda b, p: (35.0, 28.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback_senior.calc_macd", lambda c, f, s, sig: (1.0, 0.5, 0.4))
    monkeypatch.setattr("src.indicator.pullback_senior.calc_hurst_exponent", lambda c, p: 0.65)
    monkeypatch.setattr("src.indicator.pullback_senior.calc_rsi", lambda c, p: 58.0)
    monkeypatch.setattr("src.indicator.pullback_senior.calc_stochastic", lambda b, k, d: (65.0, 60.0))
    monkeypatch.setattr("src.indicator.pullback_senior.calc_bollinger_pct_b", lambda c, p: 0.65)
    monkeypatch.setattr("src.indicator.pullback_senior.calc_williams_r", lambda b, p: -30.0)

    side, reason = evaluate_senior_institutional(bars, closes, fast, slow, 1.0, optimal_hours_brt=[hour])
    assert (side, reason) == ("CALL", "setup_confirmado")
