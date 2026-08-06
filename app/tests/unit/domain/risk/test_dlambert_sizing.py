import math

import pytest

from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap
from src.domain.risk.dlambert_sizing import (
    REDIS_DLAMBERT_LINEAR_LOSSES_KEY,
    REDIS_DLAMBERT_UNIT_KEY,
    _resolve_override_value,
    dlambert_enabled,
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
    cover = 93.19 / 0.95 * 2.0
    expected = math.ceil(cover * 100) / 100
    cap = max_safe_stake_cap(10000.0, consecutive_losses_linear=3)
    expected = min(expected, cap)
    assert stake == pytest.approx(expected)


def test_resolve_dlambert_stake_ignores_last_loss_stake_for_geometric_progression():
    class RM:
        dlambert_unit = 17.89
        dlambert_config = {}
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 36.72

    payout = 0.95
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
    cover_need = 36.72 / payout * 2.0
    expected = math.ceil(cover_need * 100) / 100
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
    assert stake == pytest.approx(200.0)


def test_resolve_dlambert_stake_caps_at_bankroll_pct_and_splits_pending():
    metrics: dict = {}

    class RM:
        dlambert_unit = 100.0
        dlambert_config = {}
        soft_recovery_config = {"enabled": True, "infeasible_force_explore": False}
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
    assert stake == pytest.approx(500.0)
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is False


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


def test_resolve_dlambert_stake_zero_kelly_base_returns_zero():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        risk_params = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=0.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=100.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake == 0.0


def test_resolve_dlambert_stake_uses_rm_soft_recovery_config():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        soft_recovery_config = {"enabled": True}
        risk_params = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=10.0,
        dlambert_config={},
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=50.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake > 0.0


def test_resolve_dlambert_stake_zero_kelly_fraction_in_metrics_returns_zero():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        risk_params = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=10.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=50.0,
        dl_metrics={"kelly_fraction": 0.0},
    )
    assert tag == "D'ALEMBERT"
    assert stake == 0.0
