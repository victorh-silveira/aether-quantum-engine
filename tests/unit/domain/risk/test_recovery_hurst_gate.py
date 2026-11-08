"""Testes da trava Hurst em recovery."""

import pytest

from src.domain.risk.recovery_hurst_gate import (
    recovery_hurst_adjusted_floor,
    recovery_loss_tier_floor,
    recovery_pool_has_persistence,
    resolve_recovery_signal_floor,
)


def test_hurst_floor_unchanged_below_two_losses():
    assert recovery_hurst_adjusted_floor(0.64, 0.40, consecutive_losses=1) == 0.64


def test_hurst_floor_unchanged_when_persistent():
    assert recovery_hurst_adjusted_floor(0.64, 0.60, consecutive_losses=2) == 0.64


def test_hurst_floor_raises_when_low_hurst():
    raised = recovery_hurst_adjusted_floor(0.64, 0.45, consecutive_losses=2, log_scale=0.08)
    assert raised > 0.64


def test_recovery_pool_requires_hurst_persistence():
    candidates = [
        ("R_10", None, {"indicators": {"hurst": 0.50}}),
        ("R_10", None, {"indicators": {"hurst": 0.52}}),
    ]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=2, hurst_min=0.58) is False


def test_recovery_pool_passes_with_one_persistent_symbol():
    candidates = [
        ("R_10", None, {"indicators": {"hurst": 0.50}}),
        ("R_10", None, {"indicators": {"hurst": 0.61}}),
    ]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=2, hurst_min=0.58) is True


def test_recovery_pool_skips_check_when_losses_below_two():
    candidates = [("R_10", None, {"indicators": {"hurst": 0.40}})]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=1) is True


def test_recovery_pool_ignores_invalid_candidates():
    assert recovery_pool_has_persistence([("R_10",)], consecutive_losses=2) is False
    assert recovery_pool_has_persistence([("R_10", None, "bad")], consecutive_losses=2) is False


def test_recovery_loss_tier_floor_escalates_with_streak():
    assert recovery_loss_tier_floor(0.64, 1) == pytest.approx(0.64)
    assert recovery_loss_tier_floor(0.50, 1) == pytest.approx(0.52)
    assert recovery_loss_tier_floor(0.64, 3) >= 0.56
    assert recovery_loss_tier_floor(0.50, 5) == pytest.approx(0.58)
    assert recovery_loss_tier_floor(0.64, 0) == 0.64


def test_resolve_recovery_signal_floor_uses_log_decay():
    cfg = {
        "recovery_min_trade_score": 0.64,
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_log_scale": 0.08,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_log_decay_coef": 0.025,
        "recovery_hurst_accel_losses_min": 3,
        "recovery_hurst_severe_drawdown_min": 150.0,
    }
    floor = resolve_recovery_signal_floor(
        cfg,
        hurst=0.55,
        consecutive_losses=3,
        total_session_profit=-200.0,
        recovery_skip_counter=4,
    )
    assert floor >= 0.56
