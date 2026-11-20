"""Testes para o subsistema institucional (referencia OHLC e FVG)."""

import pytest

from src.indicator.domain import Candle
from src.indicator.institutional import detect_fvg_mitigation, session_price_reference
from src.indicator.pullback import MIN_BARS, evaluate, version_for_variant


def test_session_price_reference_calculations():
    assert session_price_reference([]) == (0.0, 0.0, 0.0)

    flat = [Candle(i * 300, 100.0, 100.0, 100.0, 100.0) for i in range(10)]
    v_flat, u_flat, l_flat = session_price_reference(flat)
    assert v_flat == pytest.approx(100.0)
    assert u_flat == pytest.approx(100.0)
    assert l_flat == pytest.approx(100.0)

    trend = [Candle(i * 300, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i) for i in range(50)]
    vwap, upper, lower = session_price_reference(trend)
    assert vwap > 0.0
    assert upper >= vwap >= lower

    short_session = [Candle(0, 100.0, 101.0, 99.0, 100.0), Candle(100000, 101.0, 102.0, 100.0, 101.0)]
    v_s, u_s, l_s = session_price_reference(short_session)
    assert (v_s, u_s, l_s) == (0.0, 0.0, 0.0)


def test_detect_fvg_mitigation():
    assert detect_fvg_mitigation([]) == (False, False)
    assert detect_fvg_mitigation([], atr=-1.0) == (False, False)

    bars_bull = [
        Candle(0, 100.0, 101.0, 99.0, 100.5),
        Candle(300, 101.0, 105.0, 100.8, 104.5),
        Candle(600, 104.5, 107.0, 103.0, 106.0),
        Candle(900, 106.0, 106.5, 101.5, 102.0),
        Candle(1200, 102.0, 105.0, 102.0, 104.8),
    ]
    bull_mit, bear_mit = detect_fvg_mitigation(bars_bull, atr=2.0)
    assert bull_mit is True

    bars_bear = [
        Candle(0, 200.0, 201.0, 199.0, 199.5),
        Candle(300, 199.0, 199.2, 195.0, 195.5),
        Candle(600, 195.5, 196.0, 192.0, 193.0),
        Candle(900, 193.0, 198.5, 192.5, 198.0),
        Candle(1200, 198.0, 198.2, 194.0, 194.5),
    ]
    bull_mit2, bear_mit2 = detect_fvg_mitigation(bars_bear, atr=2.0)
    assert bear_mit2 is True

    bars_exp_bull = [
        Candle(0, 100.0, 101.0, 99.0, 100.0),
        Candle(300, 100.0, 101.0, 99.5, 100.0),
        Candle(600, 100.0, 103.0, 99.8, 102.5),
        Candle(900, 102.5, 102.8, 100.5, 101.0),
        Candle(1200, 101.0, 103.0, 101.0, 102.8),
    ]
    exp_b, _ = detect_fvg_mitigation(bars_exp_bull, atr=1.0)
    assert exp_b is False

    bars_exp_bear = [
        Candle(0, 100.0, 101.0, 99.0, 100.0),
        Candle(300, 100.0, 101.0, 99.5, 100.0),
        Candle(600, 100.0, 100.2, 97.0, 97.5),
        Candle(900, 97.5, 99.5, 97.2, 99.0),
        Candle(1200, 99.0, 99.2, 97.0, 97.2),
    ]
    _, exp_bear = detect_fvg_mitigation(bars_exp_bear, atr=1.0)
    assert exp_bear is False


def test_evaluate_senior_institutional():
    from test_pullback import call_bars

    bars = call_bars(300)
    assert "pullback-1-" in version_for_variant("senior_institutional")

    res_inside = evaluate(
        bars,
        300,
        MIN_BARS * 300,
        variant="senior_institutional",
    )
    assert res_inside.side in ("CALL", "PUT", "SEM SINAL")


def test_senior_setup_requires_all_pillars_and_preserves_direction(monkeypatch):
    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(MIN_BARS)]
    bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    kwargs = {"variant": "senior_institutional"}
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "CALL"
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, False))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    assert evaluate(mirrored, period, MIN_BARS * period, **kwargs).side == "PUT"


def test_senior_setup_trend_continuation_flow(monkeypatch):
    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(64)]
    bars[-2] = Candle((64 - 2) * period, 101.5, 102.0, 101.4, 101.9)
    bars[-1] = Candle((64 - 1) * period, 101.9, 102.5, 101.8, 102.3)
    fast = [101.0] * 60 + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * 59 + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, False))
    kwargs = {"variant": "senior_institutional"}
    assert evaluate(bars, period, 64 * period, **kwargs).side == "CALL"

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    assert evaluate(mirrored, period, 64 * period, **kwargs).side == "PUT"


def test_senior_setup_filters_doji_adx_dmi_and_macd(monkeypatch):
    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(MIN_BARS)]
    bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    kwargs = {"variant": "senior_institutional"}

    # Doji rejection
    doji_bars = list(bars)
    doji_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 101.82)
    assert evaluate(doji_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Weak ADX rejection
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (5.0, 20.0, 18.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse DMI rejection
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse MACD rejection for CALL
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (0.0, 1.0, -1.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse MACD rejection for PUT
    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (2.0, 0.0, 2.0))
    assert evaluate(mirrored, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"


def test_senior_setup_institutional_guards(monkeypatch):
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)

    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    nine = [101.0] * (MIN_BARS - 4) + [101.6, 101.7, 101.9, 102.1]

    def mock_ema(values, span):
        if span == 9:
            return nine
        return fast if span == 20 else slow

    monkeypatch.setattr("src.indicator.pullback._ema", mock_ema)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (0.5, 0.2, 0.3))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.55)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)

    kwargs = {"variant": "senior_institutional"}

    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "CALL"

    # Guard: Hurst curto moderado (0.42-0.44) nao veta mais tendencia saudavel; Hurst extremo (< 0.38) veta
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.42)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "CALL"
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.36)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)

    # Guard: Overbought Oscillators
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 72.0)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)

    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (88.0, 80.0))
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))

    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.90)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.55)

    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -10.0)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)

    # Guard: Adverse Wick Rejection
    wick_bars = list(base_bars)
    wick_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 103.5, 101.7, 102.0)
    assert evaluate(wick_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Guard: Consecutive Bullish Streak
    streak_bars = list(base_bars)
    streak_bars[-3] = Candle((MIN_BARS - 3) * period, 101.0, 101.6, 100.9, 101.5)
    streak_bars[-2] = Candle((MIN_BARS - 2) * period, 101.5, 102.1, 101.4, 102.0)
    streak_bars[-1] = Candle((MIN_BARS - 1) * period, 102.0, 102.6, 101.9, 102.5)
    assert evaluate(streak_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Guards on PUT side
    mirrored_bars = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in base_bars]
    nine_put = [300 - x for x in nine]
    fast_put = [300 - x for x in fast]
    slow_put = [300 - x for x in slow]

    def mock_ema_put(values, span):
        if span == 9:
            return nine_put
        return fast_put if span == 20 else slow_put

    monkeypatch.setattr("src.indicator.pullback._ema", mock_ema_put)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (-0.5, -0.2, -0.3))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 48.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.45)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -55.0)

    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "PUT"

    # PUT Oversold Guard
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 28.0)
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 48.0)

    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (14.0, 20.0))
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))

    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.10)
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.45)

    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -90.0)
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -55.0)

    # PUT Adverse Lower Wick Guard
    wick_put = list(mirrored_bars)
    wick_put[-1] = Candle((MIN_BARS - 1) * period, 198.2, 198.3, 196.5, 198.0)
    assert evaluate(wick_put, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # PUT Consecutive Bearish Streak Guard
    streak_put = list(mirrored_bars)
    streak_put[-3] = Candle((MIN_BARS - 3) * period, 199.0, 199.1, 198.4, 198.5)
    streak_put[-2] = Candle((MIN_BARS - 2) * period, 198.5, 198.6, 197.9, 198.0)
    streak_put[-1] = Candle((MIN_BARS - 1) * period, 198.0, 198.1, 197.4, 197.5)
    assert evaluate(streak_put, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Climax Guards under High Momentum (CALL)
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (55.0, 30.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.60)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 68.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (94.0, 93.0))
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -5.0)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (1.0, 1.5, -0.5))
    stretched_call = list(base_bars)
    stretched_call[-1] = Candle((MIN_BARS - 1) * period, 103.5, 104.0, 103.4, 103.8)
    assert evaluate(stretched_call, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Climax Guards under High Momentum (PUT)
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (55.0, 10.0, 30.0))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.60)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 32.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (6.0, 8.0))
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -95.0)
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (4.0, 8.0))
    assert evaluate(mirrored_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -55.0)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (-1.0, -1.5, 0.5))
    stretched_put = list(mirrored_bars)
    stretched_put[-1] = Candle((MIN_BARS - 1) * period, 196.5, 196.6, 196.0, 196.2)
    assert evaluate(stretched_put, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"


def test_senior_setup_advanced_anti_loss_guards(monkeypatch):
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)

    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    nine = [101.0] * (MIN_BARS - 4) + [101.6, 101.7, 101.9, 102.1]

    def mock_ema(values, span):
        if span == 9:
            return nine
        return fast if span == 20 else slow

    monkeypatch.setattr("src.indicator.pullback._ema", mock_ema)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (0.5, 0.2, 0.3))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.55)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)

    kwargs = {"variant": "senior_institutional"}

    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "CALL"

    # 1. Guard: Declining EMA50 on CALL
    slow_declining = list(slow)
    slow_declining[-1] = slow_declining[-3] - 0.2
    monkeypatch.setattr(
        "src.indicator.pullback._ema",
        lambda values, span: nine if span == 9 else fast if span == 20 else slow_declining,
    )
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._ema", mock_ema)

    # 2. Guard: Elephant Candle (Range > 2.2 * ATR)
    giant_bars = list(base_bars)
    giant_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 104.5, 101.7, 103.5)
    assert evaluate(giant_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # 3. Guard: RSI Bearish Divergence
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 45.0 if len(c) == len(base_bars) else 55.0)
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)

    # 4. Guard: Adverse Wick Rejection (>= 45% wick and >= body)
    wick_bars = list(base_bars)
    wick_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.8, 101.75, 102.2)
    assert evaluate(wick_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"


def test_senior_setup_granular_telemetry_reasons(monkeypatch):
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)

    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (0.5, 0.2, 0.3))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.55)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)

    kwargs = {"variant": "senior_institutional"}

    bearish_conf = list(base_bars)
    bearish_conf[-1] = Candle((MIN_BARS - 1) * period, 102.2, 102.25, 101.7, 101.8)
    res = evaluate(bearish_conf, period, MIN_BARS * period, **kwargs)
    assert res.side == "SEM SINAL"
    assert res.reason == "vela_contraria"


def test_senior_setup_macd_cross_divergence_guards(monkeypatch):
    """Testa veto de PUT com MACD bullish cross e veto de CALL com MACD bearish cross."""
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)

    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.58)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 52.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (50.0, 50.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.55)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -45.0)

    kwargs = {"variant": "senior_institutional"}

    # CALL com MACD bearish cross (m_val < s_val e hist < 0): deve vetar
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (-0.2, 0.1, -0.3))
    res_call = evaluate(base_bars, period, MIN_BARS * period, **kwargs)
    assert res_call.side == "SEM SINAL"

    # PUT com MACD bullish cross (m_val > s_val e hist > 0, analogo ao DJI): deve vetar
    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in base_bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (-39.87, -50.54, 10.67))
    res_put = evaluate(mirrored, period, MIN_BARS * period, **kwargs)
    assert res_put.side == "SEM SINAL"


def test_senior_setup_operates_24h_without_killzone_veto(monkeypatch):
    """Valida que horarios anteriormente vetados (ex.: 17h00 e 00h00 BRT) passam livremente."""
    from datetime import datetime

    from src.indicator.institutional import BRT_TIMEZONE

    period = 900
    for hour in (0, 3, 11, 14, 17, 23):
        last_epoch = int(datetime(2026, 11, 10, hour, 0, tzinfo=BRT_TIMEZONE).timestamp()) - period
        start_epoch = last_epoch - (MIN_BARS - 1) * period
        bars = [Candle(start_epoch + i * period, 101.0, 102.0, 100.0, 101.0) for i in range(MIN_BARS)]
        bars[-2] = Candle(last_epoch - period, 102.0, 102.1, 101.75, 101.8)
        bars[-1] = Candle(last_epoch, 101.8, 102.3, 101.7, 102.2)
        fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
        slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
        monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span, f=fast, s=slow: f if span == 20 else s)
        monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
        monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
        monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
        res = evaluate(bars, period, last_epoch + period, variant="senior_institutional")
        assert res.side == "CALL"
        assert res.reason == "setup_confirmado"


def test_senior_setup_rejects_extreme_overbought_even_in_high_momentum(monkeypatch):
    """Testa veto de CALL com RSI >= 75 mesmo sob regime is_high_momentum."""
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)

    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    # ADX alto (72.5) e Hurst alto (0.71) ativam is_high_momentum = True
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (72.5, 32.1, 5.1))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.71)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (15.75, 11.41, 4.34))
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (82.9, 85.7))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.75)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -17.1)

    kwargs = {"variant": "senior_institutional"}

    # Com RSI = 81.4 (exaustao compradora como no GDAXI): DEVE vetar por climax_exaustao
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 81.4)
    res = evaluate(base_bars, period, MIN_BARS * period, **kwargs)
    assert res.side == "SEM SINAL"
    assert res.reason == "climax_exaustao"


def test_senior_setup_allows_mild_negative_macd_during_pullback(monkeypatch):
    """Permite recuo controlado com MACD histograma levemente negativo sem esticar da EMA20."""
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (46.0, 26.0, 9.0))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.64)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 60.0)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (70.0, 68.0))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.72)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -25.0)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (4.98, 5.64, -0.15))

    res = evaluate(base_bars, period, MIN_BARS * period, variant="senior_institutional")
    assert res.side == "SEM SINAL"
    assert res.reason == "sem_tendencia_alinhada"


def test_senior_setup_high_momentum_uniform_stoch_and_fast_dist(monkeypatch):
    """Confirma corte uniforme de Stoch 82/18 e distancia EMA20 <= 1.75 ATR sob high momentum (1.50 normal)."""
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    monkeypatch.setattr("src.indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (101.5, 105.0, 100.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (48.0, 30.0, 8.0))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.65)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 62.0)
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.80)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -20.0)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (5.0, 4.0, 1.0))

    kwargs = {"variant": "senior_institutional"}

    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (81.1, 79.0))
    assert evaluate(base_bars, period, MIN_BARS * period, **kwargs).side == "CALL"

    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (93.0, 88.0))
    res_stoch = evaluate(base_bars, period, MIN_BARS * period, **kwargs)
    assert res_stoch.side == "SEM SINAL"
    assert res_stoch.reason == "climax_exaustao"

    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (70.0, 65.0))
    within_hm = list(base_bars)
    within_hm[-1] = Candle((MIN_BARS - 1) * period, 102.8, 103.7, 102.7, 103.6)
    assert evaluate(within_hm, period, MIN_BARS * period, **kwargs).side == "CALL"

    stretched = list(base_bars)
    stretched[-1] = Candle((MIN_BARS - 1) * period, 102.8, 103.95, 102.7, 103.85)
    res_dist = evaluate(stretched, period, MIN_BARS * period, **kwargs)
    assert res_dist.side == "SEM SINAL"
    assert res_dist.reason == "afastado_da_media"

    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (22.0, 26.0, 12.0))
    assert evaluate(within_hm, period, MIN_BARS * period, **kwargs).reason == "afastado_da_media"

    frozen = list(base_bars)
    frozen[-1] = Candle((MIN_BARS - 1) * period, 102.2, 102.2, 102.2, 102.2)
    res_frozen = evaluate(frozen, period, MIN_BARS * period, **kwargs)
    assert res_frozen.side == "SEM SINAL"
    assert res_frozen.reason == "mercado_sem_oscilacao"

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in base_bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 195.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (48.0, 8.0, 30.0))
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 38.0)
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.20)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -80.0)
    monkeypatch.setattr("src.indicator.pullback._macd", lambda c, f, s, sig: (-5.0, -4.0, -1.0))
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (7.0, 12.0))
    res_put_stoch = evaluate(mirrored, period, MIN_BARS * period, **kwargs)
    assert res_put_stoch.side == "SEM SINAL"
    assert res_put_stoch.reason == "climax_exaustao"


def test_senior_setup_rejects_decelerating_macd_in_high_momentum(monkeypatch):
    """Veta PUT/CALL com MACD desacelerando mesmo sob is_high_momentum (caso OTC_FTSE 15:15)."""
    period = 900
    base_bars = [Candle(i * period, 100.0 + (i % 3) * 0.1, 100.5, 99.5, 100.0 + (i % 3) * 0.1) for i in range(MIN_BARS)]
    base_bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    base_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in base_bars]
    monkeypatch.setattr(
        "src.indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("src.indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("src.indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 195.0))
    monkeypatch.setattr("src.indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("src.indicator.pullback._adx", lambda b, p: (72.5, 5.4, 33.9))
    monkeypatch.setattr("src.indicator.pullback._hurst_exponent", lambda c, p: 0.68)
    monkeypatch.setattr("src.indicator.pullback._rsi", lambda c, p: 31.6)
    monkeypatch.setattr("src.indicator.pullback._stochastic", lambda b, k, d: (11.2, 9.3))
    monkeypatch.setattr("src.indicator.pullback._bollinger_pct_b", lambda c, p: 0.22)
    monkeypatch.setattr("src.indicator.pullback._williams_r", lambda b, p: -88.8)
    monkeypatch.setattr(
        "src.indicator.pullback._macd",
        lambda c, f, s, sig: (-13.01, -11.43, -1.58) if len(c) == len(mirrored) else (-13.9, -11.4, -2.53),
    )
    res_put = evaluate(mirrored, period, MIN_BARS * period, variant="senior_institutional")
    assert res_put.side == "SEM SINAL"
    assert res_put.reason == "sem_tendencia_alinhada"
