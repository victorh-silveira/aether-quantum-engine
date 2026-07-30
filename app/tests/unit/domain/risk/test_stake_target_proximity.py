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
    assert resolve_target_proximity_damping(101.20, 0.0) == pytest.approx(floor + span)
    assert pytest.approx(floor + span) == floor + span


def test_resolve_target_proximity_damping_at_half_target():
    target = 101.20
    pnl = target * 0.50
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(0.31)


def test_resolve_target_proximity_damping_at_ninety_percent_target():
    target = 101.20
    pnl = target * 0.90
    assert resolve_target_proximity_damping(target, pnl) == pytest.approx(0.07, abs=0.01)


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
