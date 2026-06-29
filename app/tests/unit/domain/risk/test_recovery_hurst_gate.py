"""Testes da trava Hurst em recovery."""

from src.domain.risk.recovery_hurst_gate import (
    recovery_hurst_adjusted_floor,
    recovery_pool_has_persistence,
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
        ("R_50", None, {"indicators": {"hurst": 0.50}}),
        ("R_75", None, {"indicators": {"hurst": 0.52}}),
    ]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=2, hurst_min=0.58) is False


def test_recovery_pool_passes_with_one_persistent_symbol():
    candidates = [
        ("R_50", None, {"indicators": {"hurst": 0.50}}),
        ("R_75", None, {"indicators": {"hurst": 0.61}}),
    ]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=2, hurst_min=0.58) is True


def test_recovery_pool_skips_check_when_losses_below_two():
    candidates = [("R_50", None, {"indicators": {"hurst": 0.40}})]
    assert recovery_pool_has_persistence(candidates, consecutive_losses=1) is True


def test_recovery_pool_ignores_invalid_candidates():
    assert recovery_pool_has_persistence([("R_50",)], consecutive_losses=2) is False
    assert recovery_pool_has_persistence([("R_50", None, "bad")], consecutive_losses=2) is False
