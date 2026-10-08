"""Testes unitarios para o modulo de volatilidade realizada e microestrutura."""

import math

import pytest

from src.domain.analytics.realized_volatility import (
    compute_micro_volatility_ratio,
    compute_realized_volatility,
    diagnose_micro_volatility_regime,
)


def test_compute_realized_volatility_edge_cases():
    assert compute_realized_volatility([]) == 0.0
    assert compute_realized_volatility([100.0]) == 0.0
    assert compute_realized_volatility([100.0, 100.0, 100.0]) == 0.0
    assert compute_realized_volatility([0.0, -5.0, float("nan")]) == 0.0


def test_compute_realized_volatility_known_values():
    prices = [100.0, 105.0]
    expected = math.log(105.0 / 100.0)
    assert compute_realized_volatility(prices) == pytest.approx(expected, rel=1e-6)

    prices_multi = [100.0, 105.0, 102.0]
    r1 = math.log(105.0 / 100.0)
    r2 = math.log(102.0 / 105.0)
    expected_multi = math.sqrt(r1 * r1 + r2 * r2)
    assert compute_realized_volatility(prices_multi) == pytest.approx(expected_multi, rel=1e-6)


def test_compute_micro_volatility_ratio_zero_denominator():
    assert compute_micro_volatility_ratio([100.0, 105.0], [100.0, 100.0]) == 1.0
    assert compute_micro_volatility_ratio([100.0], [100.0]) == 1.0


def test_diagnose_micro_volatility_regime_empty():
    res = diagnose_micro_volatility_regime([])
    assert res["regime"] == "normal"
    assert res["doji_risk"] is False
    assert res["tick_count_short"] == 0


def test_diagnose_micro_volatility_regime_compression_and_explosion():
    ticks_compressed = []
    base = 100.0
    for s in range(0, 240):
        base += 0.5 if s % 2 == 0 else -0.48
        ticks_compressed.append((float(s), base))
    for s in range(240, 301):
        ticks_compressed.append((float(s), base))

    res_comp = diagnose_micro_volatility_regime(ticks_compressed, current_time=300.0)
    assert res_comp["regime"] == "compression"
    assert res_comp["doji_risk"] is True
    assert res_comp["tick_count_short"] > 50

    ticks_explosion = []
    flat_px = 5000.0
    for s in range(0, 240):
        ticks_explosion.append((float(s), flat_px))
    for s in range(240, 301):
        flat_px += 5.0 if s % 2 == 0 else -4.8
        ticks_explosion.append((float(s), flat_px))

    res_exp = diagnose_micro_volatility_regime(ticks_explosion, current_time=300.0)
    assert res_exp["regime"] == "explosion"
    assert res_exp["doji_risk"] is False


def test_compute_micro_volatility_ratio_small_expected_short():
    res = compute_micro_volatility_ratio([100.0, 105.0], [100.0, 100.0 + 2e-10], time_ratio=1e-4)
    assert res == 1.0


def test_diagnose_micro_volatility_regime_normal():
    ticks_normal = []
    base_px = 100.0
    for s in range(0, 301):
        base_px += 0.1 if s % 2 == 0 else -0.09
        ticks_normal.append((float(s), base_px))

    res = diagnose_micro_volatility_regime(ticks_normal, current_time=300.0)
    assert res["regime"] == "normal"
    assert res["doji_risk"] is False
