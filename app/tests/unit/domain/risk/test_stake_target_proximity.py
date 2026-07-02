import pytest

from src.domain.risk.stake_target_proximity import (
    TARGET_DAMPING_FLOOR,
    TARGET_DAMPING_SPAN,
    apply_target_proximity_damping,
    resolve_target_proximity_damping,
)


def test_resolve_target_proximity_damping_at_session_start():
    assert resolve_target_proximity_damping(101.20, 0.0) == pytest.approx(1.0)
    assert pytest.approx(1.0) == TARGET_DAMPING_FLOOR + TARGET_DAMPING_SPAN


def test_resolve_target_proximity_damping_at_half_target():
    target = 101.20
    pnl = target * 0.50
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(0.70)


def test_resolve_target_proximity_damping_at_ninety_percent_target():
    target = 101.20
    pnl = target * 0.90
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(0.46)


def test_resolve_target_proximity_damping_at_target_floor():
    target = 101.20
    assert resolve_target_proximity_damping(target, target) == pytest.approx(TARGET_DAMPING_FLOOR)
    assert resolve_target_proximity_damping(target, target * 1.10) == pytest.approx(TARGET_DAMPING_FLOOR)


def test_apply_target_proximity_damping_scales_kelly_stake():
    target = 101.20
    raw = 45.56
    assert apply_target_proximity_damping(raw, target, 0.0) == pytest.approx(raw)
    assert apply_target_proximity_damping(raw, target, target * 0.90) == pytest.approx(raw * 0.46)
    assert apply_target_proximity_damping(raw, 0.0, 0.0) == pytest.approx(raw)


def test_target_proximity_damping_curve_on_kelly_stake():
    target = 101.20
    raw = 31.0
    at_start = apply_target_proximity_damping(raw, target, 0.0)
    at_half = apply_target_proximity_damping(raw, target, target * 0.50)
    at_ninety = apply_target_proximity_damping(raw, target, target * 0.90)
    assert at_start == pytest.approx(raw)
    assert at_half == pytest.approx(raw * resolve_target_proximity_damping(target, target * 0.50))
    assert at_ninety == pytest.approx(raw * resolve_target_proximity_damping(target, target * 0.90))
    assert at_start > at_half > at_ninety
    assert at_ninety >= raw * 0.40
