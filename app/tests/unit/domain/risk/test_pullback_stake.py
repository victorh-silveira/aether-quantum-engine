"""Invariantes monetarios do stake M5 do setup."""

import math

import pytest

from src.domain.risk.pullback_stake import pullback_stake, resolve_pullback_amort_steps


def test_base_recovery_and_residual():
    assert pullback_stake(1000, 0, 0.8, 1) == (10, 0)
    stake, covered = pullback_stake(1000, 32, 0.8, 1)
    assert stake == 40
    assert covered == 32
    stake, covered = pullback_stake(1000, 100, 0.8, 1)
    assert stake == 50
    assert covered == 40
    assert 100 - covered == 60


@pytest.mark.parametrize("available,pending,rate,minimum", [(math.nan, 0, 0.8, 1), (0, 0, 0.8, 1), (100, 0, 0.8, 6)])
def test_invalid_or_unaffordable(available, pending, rate, minimum):
    assert pullback_stake(available, pending, rate, minimum) == (0, 0)


def test_rounding_minimum_and_negative_loss():
    assert pullback_stake(123.45, -2, 0.8, 1) == (1.23, 0)
    assert pullback_stake(100, 0, 0.8, 2) == (2, 0)


def test_single_strike_target_behavior():
    assert pullback_stake(1000, 0, 0.8, 1, single_strike_target=24) == (30.0, 0.0)
    assert pullback_stake(1000, 0, 0.8, 1, single_strike_target=80) == (50.0, 0.0)
    assert pullback_stake(1000, 0, 0.8, 1, single_strike_target=5) == (10.0, 0.0)
    assert pullback_stake(1000, 16, 0.8, 1, single_strike_target=24) == (20.0, 16.0)


def test_amort_steps_smooth_recovery():
    assert pullback_stake(1000, 40, 0.8, 1, amort_steps=1) == (50.0, 40.0)
    assert pullback_stake(1000, 40, 0.8, 1, amort_steps=2) == (25.0, 20.0)


def test_resolve_pullback_amort_steps():
    assert resolve_pullback_amort_steps(0, 0.8, 50) == 1
    assert resolve_pullback_amort_steps(40, 0.8, 50) == 1
    assert resolve_pullback_amort_steps(80, 0.8, 50) == 2
    assert resolve_pullback_amort_steps(120, 0.8, 50) == 3
    assert resolve_pullback_amort_steps(math.nan, 0.8, 50) == 1
    assert resolve_pullback_amort_steps(40, 0, 50) == 1
    assert resolve_pullback_amort_steps(40, 0.8, 0) == 1
