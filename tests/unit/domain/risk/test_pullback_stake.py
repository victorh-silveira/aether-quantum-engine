"""Invariantes monetarios do stake M5 do setup."""

import math
from types import SimpleNamespace

import pytest

from src.domain.risk.pullback_stake import (
    pullback_stake,
    resolve_pullback_amort_steps,
    resolve_pullback_order_params,
    resolve_pullback_prequote_stake,
    resolve_pullback_single_strike_target,
)


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


def test_resolve_pullback_order_params_and_prequote_stake():
    rm = SimpleNamespace(initial_bankroll=8881.49, total_session_profit=0.0)
    pb_cfg = {"single_strike": True, "smooth_recovery": True}
    rm_cfg = {"large_account_stop_win_pct": 1.5, "small_account_threshold": 100.0, "params": {"stake_min": 1.0}}
    cfg = {"pullback_strategy": pb_cfg, "risk_management": rm_cfg}

    sst = resolve_pullback_single_strike_target(pb_cfg, rm_cfg, rm, 8881.49)
    assert sst == pytest.approx(133.22)
    assert resolve_pullback_single_strike_target({"single_strike": False}, rm_cfg, rm, 8881.49) == 0.0

    minimum, target, steps, cap_limit = resolve_pullback_order_params(cfg, rm, 8881.49, 400.0, 0.7246, 8881.49)
    assert minimum == 1.0
    assert target == pytest.approx(133.22)
    assert cap_limit == 444.07
    assert steps == 2

    s_first = resolve_pullback_prequote_stake(cfg, rm, 8881.49, 0.70, last_rate=0.0, unreserved=0.0)
    s_cached = resolve_pullback_prequote_stake(cfg, rm, 8881.49, 0.70, last_rate=0.7246, unreserved=0.0)
    assert s_first == pytest.approx(190.32)
    assert s_cached == pytest.approx(183.86)
