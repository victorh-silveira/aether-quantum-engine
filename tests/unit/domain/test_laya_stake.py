"""Testes unitarios para stake dinamica calibrada pelo Laya (1% a 5%)."""

import math

from src.domain.risk.laya_stake import calculate_laya_stake


def test_laya_stake_scales_linearly_between_1_and_5_percent():
    assert calculate_laya_stake(1000.0, 0.50) == (10.0, 0.01)
    assert calculate_laya_stake(1000.0, 0.75) == (30.0, 0.03)
    assert calculate_laya_stake(1000.0, 1.00) == (50.0, 0.05)
    assert calculate_laya_stake(1000.0, 0.40) == (10.0, 0.01)


def test_laya_stake_guards_invalid_or_insufficient_balance():
    assert calculate_laya_stake(math.nan, 0.80) == (0.0, 0.0)
    assert calculate_laya_stake(0.0, 0.80) == (0.0, 0.0)
    assert calculate_laya_stake(1000.0, 0.80, stake_min=0.0) == (0.0, 0.0)
    assert calculate_laya_stake(10.0, 0.80, stake_min=1.0) == (0.0, 0.0)
    assert calculate_laya_stake(100.0, 0.50, stake_min=2.0) == (2.0, 0.01)
