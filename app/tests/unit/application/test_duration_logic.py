"""Testes para a logica de duracao adaptativa."""

from src.application.services.llm.duration_logic import calculate_adaptive_duration, enforce_minimum_duration


def test_calculate_adaptive_duration_defaults():
    assert calculate_adaptive_duration({}, {}, base_duration=2) == 2


def test_calculate_adaptive_duration_regime():
    assert calculate_adaptive_duration({}, {"regime_label": "range"}, base_duration=2) == 2
    assert calculate_adaptive_duration({}, {"regime_label": "trend_fraca"}, base_duration=2) == 2
    assert calculate_adaptive_duration({}, {"regime_label": "trend_forte"}, base_duration=2) == 2


def test_calculate_adaptive_duration_volatility():
    assert calculate_adaptive_duration({}, {"atr_m5_pct": 0.19}, base_duration=2) == 2
    assert calculate_adaptive_duration({}, {"atr_m5_pct": 0.30}, base_duration=2) == 3
    assert calculate_adaptive_duration({}, {"atr_m5_pct": 0.45}, base_duration=2) == 5


def test_calculate_adaptive_duration_cap():
    assert calculate_adaptive_duration({}, {"atr_m5_pct": 1.0}, base_duration=2) == 5


def test_calculate_adaptive_duration_bypass_m1():
    assert calculate_adaptive_duration({}, {"regime_label": "range", "atr_m5_pct": 1.0}, base_duration=1) == 1


def test_calculate_adaptive_duration_multiplier():
    assert calculate_adaptive_duration({}, {}, base_duration="MULT") == "MULT"


def test_enforce_minimum_duration():
    assert enforce_minimum_duration("frxEURUSD", 15) == 15
    assert enforce_minimum_duration("OTC_GDAXI", 15) == 15
