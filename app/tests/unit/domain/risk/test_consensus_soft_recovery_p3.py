import pytest

from src.domain.risk.consensus_stake_penalty import (
    adaptive_recovery_progression_factor,
    apply_soft_recovery_stake,
    max_safe_stake_cap,
    soft_recovery_progression_multiplier,
)


def test_sequential_drawdown_stakes_use_amort_cover():
    bankroll = 11000.0
    unit = 17.25
    payout = 0.95
    pending = 50.0
    soft = {
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "cover_multiple": 1.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stakes = [
        apply_soft_recovery_stake(
            pending_total=pending,
            base_unit=unit,
            consecutive_losses=losses,
            previous_stake=0.0,
            bankroll=bankroll,
            payout=payout,
            soft_recovery=soft,
            metrics={},
        )
        for losses in (1, 2, 3, 4, 5)
    ]
    assert stakes[0] == pytest.approx(pending / payout / 4.0)
    assert stakes[1] == pytest.approx(pending / payout / 3.0)
    assert stakes[-1] == pytest.approx(pending / payout / 2.0)
    assert stakes[-1] <= max_safe_stake_cap(bankroll, consecutive_losses_linear=5, soft_recovery=soft)


def test_adaptive_recovery_factor_at_payout_ninety_five():
    assert adaptive_recovery_progression_factor(0.95) == pytest.approx(2.0526315789)


def test_adaptive_recovery_factor_expands_at_payout_seventy_five():
    assert adaptive_recovery_progression_factor(0.75) == pytest.approx(2.3333333333)


def test_adaptive_recovery_factor_expands_at_payout_seventy():
    assert adaptive_recovery_progression_factor(0.70) == pytest.approx(2.4285714286)


def test_adaptive_recovery_factor_caps_at_two_fifty():
    assert adaptive_recovery_progression_factor(0.40) == pytest.approx(2.50)


def test_soft_recovery_progression_multiplier_scales_with_payout_stress():
    high = soft_recovery_progression_multiplier(2, payout=0.95)
    low = soft_recovery_progression_multiplier(2, payout=0.70)
    assert low >= high


def test_apply_soft_recovery_stake_expands_when_payout_degrades():
    base_kwargs = {
        "pending_total": 80.0,
        "base_unit": 15.0,
        "consecutive_losses": 1,
        "previous_stake": 0.0,
        "bankroll": 11500.0,
    }
    stake_hi = apply_soft_recovery_stake(**base_kwargs, payout=0.95, metrics={})
    stake_lo = apply_soft_recovery_stake(**base_kwargs, payout=0.70, metrics={})
    assert stake_lo > stake_hi
