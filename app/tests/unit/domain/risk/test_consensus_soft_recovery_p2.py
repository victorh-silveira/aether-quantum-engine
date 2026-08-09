import pytest

from src.domain.risk.consensus_stake_penalty import (
    apply_soft_recovery_stake,
    max_safe_stake_cap,
)


def test_large_bankroll_max_safe_stake_uses_pct_not_abs_cap():
    soft = {
        "max_safe_stake_cap": 4.20,
        "max_safe_stake_pct": 0.05,
        "max_safe_stake_pct_linear2": 0.025,
        "max_safe_stake_pct_linear3": 0.020,
    }
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=0, soft_recovery=soft) == pytest.approx(600.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=1, soft_recovery=soft) == pytest.approx(600.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=2, soft_recovery=soft) == pytest.approx(300.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=3, soft_recovery=soft) == pytest.approx(240.0)


def test_apply_soft_recovery_stake_dampens_on_indicator_hurst():
    metrics = {"indicators": {"hurst": 0.35}, "regime_chop_soft": False}
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=15.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        payout=0.95,
        metrics=metrics,
    )
    cover = 80.0 / 0.95 * 2.0
    assert stake < cover
    assert stake == pytest.approx(28.75)
    assert metrics.get("recovery_low_hurst_damped") is True
    assert metrics.get("recovery_force_explore") is True


def test_apply_soft_recovery_stake_dampens_on_chop_neg_edge():
    metrics = {
        "indicators": {"hurst": 0.48},
        "regime_chop_soft": True,
        "neg_edge_soft": True,
    }
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=15.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        payout=0.95,
        metrics=metrics,
    )
    cover = 80.0 / 0.95 * 2.0
    assert stake < cover
    assert stake == pytest.approx(28.75)
    assert metrics.get("recovery_chop_neg_edge_damped") is True
    assert metrics.get("recovery_force_explore") is True


def test_apply_soft_recovery_stake_dampens_on_neg_edge_alone():
    metrics = {
        "indicators": {"hurst": 0.55},
        "regime_chop_soft": False,
        "neg_edge_soft": True,
    }
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=15.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        payout=0.95,
        metrics=metrics,
    )
    cover = 80.0 / 0.95 * 2.0
    assert stake < cover
    assert stake == pytest.approx(28.75)
    assert metrics.get("recovery_chop_neg_edge_damped") is True
    assert metrics.get("recovery_force_explore") is True


def test_small_account_hard_floor_caps_recovery_at_five_percent():
    soft = {"max_safe_stake_cap": 4.20}
    assert max_safe_stake_cap(80.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(75.0, consecutive_losses_linear=5, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.20)
