import pytest

from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings
from src.domain.risk.stake_target_proximity import (
    apply_target_proximity_damping,
    resolve_target_proximity_damping,
)


def _damping_knobs():
    runtime = load_kelly_runtime_from_settings()
    return float(runtime["target_damping_floor"]), float(runtime["target_damping_span"])


def test_resolve_target_proximity_damping_at_session_start():
    floor, span = _damping_knobs()
    assert floor == pytest.approx(0.70)
    assert span == pytest.approx(0.30)
    assert resolve_target_proximity_damping(101.20, 0.0) == pytest.approx(1.0)
    assert resolve_target_proximity_damping(101.20, 0.0) == pytest.approx(floor + span)


def test_resolve_target_proximity_damping_at_half_target():
    floor, span = _damping_knobs()
    target = 101.20
    pnl = target * 0.50
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(floor + span * 0.50)


def test_resolve_target_proximity_damping_at_ninety_percent_target():
    floor, span = _damping_knobs()
    target = 101.20
    pnl = target * 0.90
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(floor + span * 0.10, abs=0.01)


def test_resolve_target_proximity_damping_at_target_floor():
    floor, _span = _damping_knobs()
    target = 101.20
    assert resolve_target_proximity_damping(target, target) == pytest.approx(floor)
    assert resolve_target_proximity_damping(target, target * 1.10) == pytest.approx(floor)


def test_apply_target_proximity_damping_scales_kelly_stake():
    target = 101.20
    raw = 45.56
    assert apply_target_proximity_damping(raw, target, 0.0) == pytest.approx(
        raw * resolve_target_proximity_damping(target, 0.0)
    )
    assert apply_target_proximity_damping(raw, target, target * 0.90) == pytest.approx(
        raw * resolve_target_proximity_damping(target, target * 0.90)
    )
    assert apply_target_proximity_damping(raw, 0.0, 0.0) == pytest.approx(raw)


def test_target_proximity_damping_curve_on_kelly_stake():
    target = 101.20
    raw = 31.0
    at_start = apply_target_proximity_damping(raw, target, 0.0)
    at_half = apply_target_proximity_damping(raw, target, target * 0.50)
    at_ninety = apply_target_proximity_damping(raw, target, target * 0.90)
    assert at_start == pytest.approx(raw * resolve_target_proximity_damping(target, 0.0))
    assert at_half == pytest.approx(raw * resolve_target_proximity_damping(target, target * 0.50))
    assert at_ninety == pytest.approx(raw * resolve_target_proximity_damping(target, target * 0.90))
    assert at_start > at_half > at_ninety


def test_apply_target_proximity_to_kelly_caps_at_needed_stake():
    from types import SimpleNamespace

    from src.domain.risk.risk_stake_flow import apply_target_proximity_to_kelly

    rm = SimpleNamespace(
        config={"risk_management": {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.0431}}},
        initial_bankroll=10000.0,
        total_session_profit=350.0,
        kelly_config={},
        risk_params={"stake_min": 1.0},
    )
    capped = apply_target_proximity_to_kelly(rm, 500.0, apply_stop_win=True, payout=0.85)
    assert capped < 110.0
    assert capped >= 95.0


def test_apply_session_profit_lock_inactive_when_target_zero():
    from src.domain.risk.stake_target_proximity import apply_session_profit_lock

    assert apply_session_profit_lock(100.0, 0.0, 50.0, 50.0) == 100.0


def test_apply_session_profit_lock_inactive_when_peak_below_half_target():
    from src.domain.risk.stake_target_proximity import apply_session_profit_lock

    target = 400.0
    assert apply_session_profit_lock(100.0, target, 150.0, 180.0) == 100.0


def test_apply_session_profit_lock_caps_stake_to_protect_locked_profit():
    from src.domain.risk.stake_target_proximity import apply_session_profit_lock

    target = 400.0
    peak = 220.0
    pnl = 130.0
    capped = apply_session_profit_lock(120.0, target, pnl, peak, stake_min=1.0)
    assert capped == pytest.approx(30.0)


def test_apply_session_profit_lock_floors_at_stake_min_when_at_or_below_locked():
    from src.domain.risk.stake_target_proximity import apply_session_profit_lock

    target = 400.0
    peak = 220.0
    pnl = 95.0
    capped = apply_session_profit_lock(120.0, target, pnl, peak, stake_min=1.5)
    assert capped == pytest.approx(1.5)
