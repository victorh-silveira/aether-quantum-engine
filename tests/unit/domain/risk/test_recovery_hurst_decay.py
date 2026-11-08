import math

import pytest

from src.domain.risk.recovery_hurst_decay import (
    effective_recovery_hurst_min,
    resolve_effective_hurst_min,
    session_drawdown_from_profit,
)


def test_effective_recovery_hurst_min_decay_to_floor():
    assert effective_recovery_hurst_min(0.58, 0) == 0.58
    assert effective_recovery_hurst_min(0.58, 3) == pytest.approx(0.55)
    assert effective_recovery_hurst_min(0.58, 8) == 0.50
    assert effective_recovery_hurst_min(0.58, 20) == 0.50


def test_log_decay_accelerates_when_losses_and_drawdown_severe():
    linear = effective_recovery_hurst_min(0.58, 4, decay=0.01)
    accelerated = effective_recovery_hurst_min(
        0.58,
        4,
        consecutive_losses=3,
        session_drawdown=200.0,
        severe_drawdown_min=150.0,
        log_decay_coef=0.025,
    )
    assert accelerated < linear
    assert accelerated == pytest.approx(0.58 - 0.025 * math.log1p(4))


def test_log_decay_not_used_when_drawdown_low():
    linear = effective_recovery_hurst_min(0.58, 4, decay=0.01)
    mild = effective_recovery_hurst_min(
        0.58,
        4,
        consecutive_losses=3,
        session_drawdown=50.0,
        severe_drawdown_min=150.0,
    )
    assert mild == linear


def test_session_drawdown_from_profit():
    assert session_drawdown_from_profit(100.0) == 0.0
    assert session_drawdown_from_profit(-175.5) == pytest.approx(175.5)


def test_resolve_effective_hurst_min_with_accel_config():
    cfg = {
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_log_decay_coef": 0.025,
        "recovery_hurst_accel_losses_min": 3,
        "recovery_hurst_severe_drawdown_min": 150.0,
    }
    val = resolve_effective_hurst_min(
        cfg,
        4,
        consecutive_losses=3,
        session_drawdown=200.0,
    )
    linear = resolve_effective_hurst_min(cfg, 4)
    assert val < linear


def test_resolve_effective_hurst_min_disabled():
    cfg = {
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_decay_enabled": False,
    }
    assert resolve_effective_hurst_min(cfg, 10) == 0.58
