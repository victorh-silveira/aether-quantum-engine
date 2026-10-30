"""Testes para o subsistema institucional (VWAP, FVG e Killzones)."""

import pytest

from indicator.domain import Candle
from indicator.institutional import detect_fvg_mitigation, is_within_killzone, session_price_reference
from indicator.pullback import MIN_BARS, evaluate, version_for_variant


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


def test_is_within_killzone():
    assert is_within_killzone(1700000000, []) is False

    from datetime import datetime

    from indicator.institutional import BRT_TIMEZONE

    epoch = 1700000000
    hour = datetime.fromtimestamp(epoch, tz=BRT_TIMEZONE).hour
    assert is_within_killzone(epoch, [hour]) is True
    assert is_within_killzone(epoch, [(hour + 1) % 24]) is False


def test_evaluate_senior_institutional():
    from test_pullback import call_bars

    bars = call_bars(300)
    assert "pullback-1-" in version_for_variant("senior_institutional")

    res_outside = evaluate(
        bars,
        300,
        MIN_BARS * 300,
        variant="senior_institutional",
        optimal_hours_brt=[99],
    )
    assert res_outside.reason == "fora_da_killzone_brt"

    res_inside = evaluate(
        bars,
        300,
        MIN_BARS * 300,
        variant="senior_institutional",
        optimal_hours_brt=None,
    )
    assert res_inside.side in ("CALL", "PUT", "SEM SINAL")


def test_senior_setup_requires_all_pillars_and_preserves_direction(monkeypatch):
    from datetime import datetime

    from indicator.institutional import BRT_TIMEZONE

    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(MIN_BARS)]
    bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    hour = datetime.fromtimestamp(bars[-1].epoch, tz=BRT_TIMEZONE).hour
    kwargs = {"variant": "senior_institutional", "optimal_hours_brt": [hour]}
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "CALL"
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, False))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
    assert evaluate(bars, period, MIN_BARS * period, variant="senior_institutional", optimal_hours_brt=[]).reason == (
        "fora_da_killzone_brt"
    )

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    assert evaluate(mirrored, period, MIN_BARS * period, **kwargs).side == "PUT"


def test_senior_setup_trend_continuation_flow(monkeypatch):
    from datetime import datetime

    from indicator.institutional import BRT_TIMEZONE

    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(64)]
    bars[-2] = Candle((64 - 2) * period, 101.5, 102.0, 101.4, 101.9)
    bars[-1] = Candle((64 - 1) * period, 101.9, 102.5, 101.8, 102.3)
    fast = [101.0] * 60 + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * 59 + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, False))
    hour = datetime.fromtimestamp(bars[-1].epoch, tz=BRT_TIMEZONE).hour
    kwargs = {"variant": "senior_institutional", "optimal_hours_brt": [hour]}
    assert evaluate(bars, period, 64 * period, **kwargs).side == "CALL"

    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    assert evaluate(mirrored, period, 64 * period, **kwargs).side == "PUT"


def test_senior_setup_filters_doji_adx_dmi_and_macd(monkeypatch):
    from datetime import datetime

    from indicator.institutional import BRT_TIMEZONE

    period = 900
    bars = [Candle(i * period, 101.0, 102.0, 100.0, 101.0) for i in range(MIN_BARS)]
    bars[-2] = Candle((MIN_BARS - 2) * period, 102.0, 102.1, 101.75, 101.8)
    bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 102.2)
    fast = [101.0] * (MIN_BARS - 4) + [101.5, 101.6, 101.8, 102.0]
    slow = [100.0] * (MIN_BARS - 5) + [100.0, 100.1, 100.2, 100.3, 100.4]
    monkeypatch.setattr("indicator.pullback._ema", lambda values, span: fast if span == 20 else slow)
    monkeypatch.setattr("indicator.pullback._atr", lambda values: 1.0)
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (101.5, 103.0, 100.0))
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (True, False))
    hour = datetime.fromtimestamp(bars[-1].epoch, tz=BRT_TIMEZONE).hour
    kwargs = {"variant": "senior_institutional", "optimal_hours_brt": [hour]}

    # Doji rejection
    doji_bars = list(bars)
    doji_bars[-1] = Candle((MIN_BARS - 1) * period, 101.8, 102.3, 101.7, 101.82)
    assert evaluate(doji_bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Weak ADX rejection
    monkeypatch.setattr("indicator.pullback._adx", lambda b, p: (5.0, 20.0, 18.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse DMI rejection
    monkeypatch.setattr("indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse MACD rejection for CALL
    monkeypatch.setattr("indicator.pullback._adx", lambda b, p: (25.0, 25.0, 10.0))
    monkeypatch.setattr("indicator.pullback._macd", lambda c, f, s, sig: (0.0, 1.0, -1.0))
    assert evaluate(bars, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"

    # Adverse MACD rejection for PUT
    mirrored = [Candle(b.epoch, 300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close) for b in bars]
    monkeypatch.setattr(
        "indicator.pullback._ema", lambda values, span: [300 - x for x in (fast if span == 20 else slow)]
    )
    monkeypatch.setattr("indicator.pullback.session_price_reference", lambda values: (198.5, 200.0, 197.0))
    monkeypatch.setattr("indicator.pullback.detect_fvg_mitigation", lambda values, atr: (False, True))
    monkeypatch.setattr("indicator.pullback._adx", lambda b, p: (25.0, 10.0, 25.0))
    monkeypatch.setattr("indicator.pullback._macd", lambda c, f, s, sig: (2.0, 0.0, 2.0))
    assert evaluate(mirrored, period, MIN_BARS * period, **kwargs).side == "SEM SINAL"
