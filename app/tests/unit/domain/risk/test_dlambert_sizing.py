import math

import pytest

from src.domain.risk.dlambert_sizing import (
    REDIS_DLAMBERT_LINEAR_LOSSES_KEY,
    REDIS_DLAMBERT_UNIT_KEY,
    _resolve_override_value,
    dlambert_enabled,
    dlambert_log_suffix,
    effective_soft_recovery_base,
    resolve_dlambert_stake,
    resolve_dlambert_unit,
    soft_recovery_stress_active,
)


def test_redis_keys():
    assert REDIS_DLAMBERT_UNIT_KEY == "session:current:dlambert_unit"
    assert REDIS_DLAMBERT_LINEAR_LOSSES_KEY == "session:current:consecutive_losses_linear"


def test_dlambert_enabled_defaults_true_and_respects_flag():
    assert dlambert_enabled({}) is True
    assert dlambert_enabled({"dlambert_enabled": False}) is False


def test_resolve_dlambert_unit_captures_first_kelly():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": None}

    rm = RM()
    assert resolve_dlambert_unit(42.5, rm) == pytest.approx(42.5)
    assert rm.dlambert_unit == pytest.approx(42.5)
    assert resolve_dlambert_unit(99.0, rm) == pytest.approx(42.5)


def test_resolve_dlambert_unit_override():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": 30.0}

    rm = RM()
    assert resolve_dlambert_unit(50.0, rm) == pytest.approx(30.0)


def test_resolve_dlambert_unit_invalid_override():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": "bad"}

    rm = RM()
    assert resolve_dlambert_unit(25.0, rm) == pytest.approx(25.0)


def test_resolve_dlambert_unit_zero_kelly_returns_zero():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    assert resolve_dlambert_unit(0.0, RM()) == pytest.approx(0.0)


def test_resolve_dlambert_stake_kelly_when_flat():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=10000.0,
        kelly_base=55.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(55.0)


def test_resolve_override_value_handles_invalid_and_non_positive_values():
    assert _resolve_override_value({}) == pytest.approx(0.0)
    assert _resolve_override_value({"dlambert_unit_override": "bad"}) == pytest.approx(0.0)
    assert _resolve_override_value({"dlambert_unit_override": -2.0}) == pytest.approx(0.0)


def test_soft_recovery_stress_active_pending_or_linear_losses():
    assert soft_recovery_stress_active(recovery_active=False, pending_total=93.19, consecutive_losses_linear=0)
    assert soft_recovery_stress_active(recovery_active=False, pending_total=0.0, consecutive_losses_linear=3)
    assert not soft_recovery_stress_active(recovery_active=False, pending_total=0.0, consecutive_losses_linear=0)


def test_effective_soft_recovery_base_anchors_above_compressed_kelly():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {"dlambert_unit_override": 8.0}

    assert effective_soft_recovery_base(1.0, RM(), {"dlambert_unit_override": 8.0}) == pytest.approx(10.0)


def test_resolve_dlambert_stake_soft_recovery_with_progression():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=10000.0,
        kelly_base=1.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=93.19,
        payout=0.95,
    )
    assert tag == "D'ALEMBERT"
    session_unit = max(10.0, 10000.0 * 0.0015)
    cover = 93.19 / 0.95 / 2.0
    expected = math.ceil(max(session_unit * 1.15, cover) * 100) / 100
    assert stake == pytest.approx(expected)


def test_resolve_dlambert_stake_ignores_last_loss_stake_for_geometric_progression():
    class RM:
        dlambert_unit = 17.89
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 36.72

    payout = 0.95
    factor = 1.0 + (1.0 / payout)
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=11926.67,
        kelly_base=1.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=1,
        pending_total=36.72,
        payout=payout,
    )
    assert tag == "D'ALEMBERT"
    geometric = 17.89 * factor
    cover_full = 36.72 / payout
    amort_cycles = max(2, 5 - min(1, 3))
    cover_need = cover_full / float(amort_cycles)
    expected = math.ceil(max(geometric, cover_need) * 100) / 100
    assert stake == pytest.approx(expected)


def test_resolve_dlambert_stake_linear_losses_without_pending_use_base_unit():
    class RM:
        dlambert_unit = 50.0
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=1.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=0.0,
        payout=0.95,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(50.0)


def test_resolve_dlambert_stake_caps_at_bankroll_pct_and_splits_pending():
    metrics: dict = {}

    class RM:
        dlambert_unit = 100.0
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=100.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=8,
        pending_total=5000.0,
        payout=0.95,
        dl_metrics=metrics,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(200.0)


def test_resolve_dlambert_stake_falls_back_to_kelly_when_disabled():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}
        risk_params = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=40.0,
        dlambert_config={"dlambert_enabled": False},
        rm=RM(),
        consecutive_losses_linear=4,
        pending_total=200.0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(40.0)


def test_dlambert_log_suffix_soft_recovery_and_empty():
    suffix = dlambert_log_suffix(
        "D'ALEMBERT",
        108.62,
        93.19,
        10.0,
        consecutive_losses_linear=2,
        payout=0.95,
    )
    assert "soft=2.05x^2" in suffix
    assert "p=0.95" in suffix
    assert "U=$10.00" in suffix
    fixed = dlambert_log_suffix(
        "D'ALEMBERT",
        17.25,
        6.75,
        15.0,
        consecutive_losses_linear=3,
        payout=0.95,
    )
    assert "fixed=U+15%" in fixed
    assert "n=3" in fixed
    assert dlambert_log_suffix("KELLY", 55.0, 0.0, 55.0) == ""
