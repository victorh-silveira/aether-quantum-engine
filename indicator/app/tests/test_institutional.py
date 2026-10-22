"""Testes para o subsistema institucional (VWAP, FVG e Killzones)."""

import pytest

from indicator.domain import Candle
from indicator.institutional import detect_fvg_mitigation, is_within_killzone, session_vwap
from indicator.pullback import MIN_BARS, evaluate, version_for_variant


def test_session_vwap_calculations():
    assert session_vwap([]) == (0.0, 0.0, 0.0)

    flat = [Candle(i * 300, 100.0, 100.0, 100.0, 100.0) for i in range(10)]
    v_flat, u_flat, l_flat = session_vwap(flat)
    assert v_flat == pytest.approx(100.0)
    assert u_flat == pytest.approx(100.0)
    assert l_flat == pytest.approx(100.0)

    trend = [Candle(i * 300, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i) for i in range(50)]
    vwap, upper, lower = session_vwap(trend)
    assert vwap > 0.0
    assert upper >= vwap >= lower


def test_detect_fvg_mitigation():
    assert detect_fvg_mitigation([]) == (True, True)
    assert detect_fvg_mitigation([], atr=-1.0) == (True, True)

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
    assert exp_b is True

    bars_exp_bear = [
        Candle(0, 100.0, 101.0, 99.0, 100.0),
        Candle(300, 100.0, 101.0, 99.5, 100.0),
        Candle(600, 100.0, 100.2, 97.0, 97.5),
        Candle(900, 97.5, 99.5, 97.2, 99.0),
        Candle(1200, 99.0, 99.2, 97.0, 97.2),
    ]
    _, exp_bear = detect_fvg_mitigation(bars_exp_bear, atr=1.0)
    assert exp_bear is True


def test_is_within_killzone():
    assert is_within_killzone(1700000000, None) is True
    assert is_within_killzone(1700000000, []) is True

    from datetime import datetime

    from indicator.institutional import BRT_TIMEZONE

    epoch = 1700000000
    hour = datetime.fromtimestamp(epoch, tz=BRT_TIMEZONE).hour
    assert is_within_killzone(epoch, [hour]) is True
    assert is_within_killzone(epoch, [(hour + 1) % 24]) is False


def test_evaluate_senior_institutional():
    from test_pullback import call_bars

    bars = call_bars(300)
    assert "senior-1-" in version_for_variant("senior_institutional")

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
