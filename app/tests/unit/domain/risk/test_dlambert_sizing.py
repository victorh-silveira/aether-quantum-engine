import pytest

from src.domain.risk.dlambert_sizing import (
    BOOSTER_DAMPING_FACTOR,
    REDIS_DLAMBERT_LINEAR_LOSSES_KEY,
    REDIS_DLAMBERT_UNIT_KEY,
    dlambert_amortization_multiplier,
    dlambert_recovery_stake,
    effective_dlambert_unit,
    resolve_dlambert_stake,
    resolve_dlambert_unit,
)


def test_redis_keys():
    assert REDIS_DLAMBERT_UNIT_KEY == "session:current:dlambert_unit"
    assert REDIS_DLAMBERT_LINEAR_LOSSES_KEY == "session:current:consecutive_losses_linear"


def test_dlambert_recovery_stake_linear():
    assert dlambert_recovery_stake(50.0, 25.0, 2) == pytest.approx(100.0)


def test_amortization_booster_scales_unit_with_deep_drawdown():
    bankroll = 10000.0
    pending = 400.0
    unit = 20.0
    assert dlambert_amortization_multiplier(pending, bankroll) == pytest.approx(1.75)
    assert effective_dlambert_unit(unit, pending, bankroll) == pytest.approx(35.0)
    assert dlambert_recovery_stake(
        50.0,
        unit,
        2,
        pending_total=pending,
        bankroll=bankroll,
    ) == pytest.approx(120.0)


def test_amortization_booster_caps_multiplier_at_one_point_seven_five():
    bankroll = 10000.0
    pending = 800.0
    assert dlambert_amortization_multiplier(pending, bankroll) == pytest.approx(1.75)
    assert effective_dlambert_unit(10.0, pending, bankroll) == pytest.approx(17.5)


def test_booster_damping_softens_c0005_drawdown_stake():
    bankroll = 10000.0
    pending = 141.78
    unit = 35.0
    kelly_base = 50.0
    linear = 2
    u_eff_damped = effective_dlambert_unit(unit, pending, bankroll)
    u_eff_undamped = effective_dlambert_unit(unit, pending, bankroll, damping=1.0)
    damped_stake = kelly_base + linear * u_eff_damped
    undamped_stake = kelly_base + linear * u_eff_undamped
    assert damped_stake < undamped_stake
    assert damped_stake < 175.23
    assert pytest.approx(0.50) == BOOSTER_DAMPING_FACTOR


def test_amortization_floor_secondary_symbol_deep_drawdown():
    bankroll = 30000.0
    pending = 653.12
    unit = 14.30
    ratio = min(1.5, pending / (bankroll * 0.02))
    expected_u_eff = unit * (1.0 + ratio * BOOSTER_DAMPING_FACTOR)
    assert effective_dlambert_unit(unit, pending, bankroll) == pytest.approx(expected_u_eff)
    stake = dlambert_recovery_stake(
        8.0,
        unit,
        1,
        pending_total=pending,
        bankroll=bankroll,
    )
    assert stake == pytest.approx(8.0 + expected_u_eff)
    assert stake < 8.0 + unit * 2.5


def test_amortization_multiplier_returns_one_without_drawdown():
    assert dlambert_amortization_multiplier(0.0, 10000.0) == pytest.approx(1.0)
    assert dlambert_amortization_multiplier(100.0, 0.0) == pytest.approx(1.0)


def test_amortization_below_drawdown_gate_keeps_progressive_scaling():
    bankroll = 10000.0
    pending = 150.0
    unit = 20.0
    assert pending <= bankroll * 0.02
    assert effective_dlambert_unit(unit, pending, bankroll) == pytest.approx(
        unit * dlambert_amortization_multiplier(pending, bankroll)
    )


def test_resolve_dlambert_stake_applies_booster_without_ceiling():
    class RM:
        dlambert_unit = 20.0
        dlambert_config = {}

    cfg = {"dlambert_enabled": True}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=50.0,
        dlambert_config=cfg,
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=400.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(120.0)


def test_resolve_dlambert_stake_progresses_without_bankroll_cap():
    class RM:
        dlambert_unit = 500.0
        dlambert_config = {}

    cfg = {"dlambert_enabled": True}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=100.0,
        dlambert_config=cfg,
        rm=RM(),
        consecutive_losses_linear=5,
        pending_total=800.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake > 10000.0 * 0.04


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


def test_resolve_dlambert_stake_kelly_vs_recovery():
    class RM:
        dlambert_unit = 20.0
        dlambert_config = {}

    cfg = {"dlambert_enabled": True}
    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=10000.0,
        kelly_base=55.0,
        dlambert_config=cfg,
        rm=RM(),
        consecutive_losses_linear=0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(55.0)

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=50.0,
        dlambert_config=cfg,
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=0.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(90.0)
