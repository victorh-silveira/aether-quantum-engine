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


def test_apply_soft_recovery_stake_cover_despite_low_hurst_when_pending():
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
    cover = 80.0 / 0.95 / 1.0 * 1.5
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_force_explore") is False
    assert metrics.get("recovery_cover_need") == pytest.approx(cover)


def test_apply_soft_recovery_stake_cover_despite_chop_neg_edge_when_pending():
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
    cover = 80.0 / 0.95 / 1.0 * 1.5
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_force_explore") is False


def test_apply_soft_recovery_stake_cover_despite_neg_edge_alone_when_pending():
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
    cover = 80.0 / 0.95 / 1.0 * 1.5
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_force_explore") is False


def test_neg_edge_sticky_unit_110_uses_cover_when_pending():
    metrics = {"neg_edge_soft": True}
    soft = {
        "material_pending_min": 0.25,
        "cover_multiple": 1.5,
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=90.0,
        base_unit=110.48,
        consecutive_losses=1,
        previous_stake=110.48,
        bankroll=11000.0,
        payout=0.95,
        metrics=metrics,
        soft_recovery=soft,
    )
    cover = 90.0 / 0.95 / 1.0 * 1.5
    floor = 11000.0 * 0.0025
    assert stake == pytest.approx(cover)
    assert stake > floor
    assert metrics.get("recovery_force_explore") is False
    assert metrics.get("recovery_cover_need") == pytest.approx(cover)


def test_neg_edge_without_pending_uses_neutral_floor_not_sticky_u():
    metrics = {"neg_edge_soft": True}
    soft = {"material_pending_min": 0.25, "max_safe_stake_pct": 0.05}
    stake = apply_soft_recovery_stake(
        pending_total=0.0,
        base_unit=110.48,
        consecutive_losses=0,
        previous_stake=110.48,
        bankroll=11000.0,
        payout=0.95,
        metrics=metrics,
        soft_recovery=soft,
    )
    assert stake == pytest.approx(27.5)
    assert metrics.get("recovery_explore_used_cover") is False
    assert metrics.get("recovery_force_explore_reason") == "neg_edge"
    assert metrics.get("recovery_force_explore") is True


def test_infeasible_force_explore_sticky_u_uses_neutral_floor():
    metrics: dict = {}
    soft = {
        "enabled": True,
        "max_safe_stake_pct": 0.025,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "infeasible_force_explore": True,
        "material_pending_min": 0.25,
    }
    stake = apply_soft_recovery_stake(
        pending_total=800.0,
        base_unit=110.48,
        consecutive_losses=2,
        previous_stake=110.48,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
    )
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_force_explore_reason") == "infeasible"
    assert stake == pytest.approx(25.0)
    assert metrics.get("recovery_explore_used_cover") is False


def test_dal_normal_pending_without_neg_edge_keeps_cover():
    metrics = {"indicators": {"hurst": 0.55}}
    soft = {
        "material_pending_min": 0.25,
        "cover_multiple": 1.5,
        "amort_cycles_min": 2,
        "amort_cycles_max": 4,
        "max_safe_stake_pct": 0.05,
        "infeasible_force_explore": True,
        "near_stop_win_freeze_pct": 0.99,
    }
    stake = apply_soft_recovery_stake(
        pending_total=90.0,
        base_unit=110.48,
        consecutive_losses=1,
        previous_stake=50.0,
        bankroll=11000.0,
        payout=0.95,
        metrics=metrics,
        soft_recovery=soft,
    )
    amort = 3
    cover = 90.0 / 0.95 / float(amort) * 1.5
    assert metrics.get("recovery_force_explore") is False
    assert stake == pytest.approx(cover, rel=1e-3)


def test_neg_edge_large_pending_sets_infeasible_with_floor():
    metrics = {"neg_edge_soft": True}
    soft = {
        "material_pending_min": 0.25,
        "cover_multiple": 1.5,
        "amort_cycles_min": 2,
        "amort_cycles_max": 4,
        "max_safe_stake_pct": 0.025,
        "max_safe_stake_pct_linear3": 0.025,
        "infeasible_force_explore": True,
        "near_stop_win_freeze_pct": 0.99,
    }
    stake = apply_soft_recovery_stake(
        pending_total=1026.50,
        base_unit=110.0,
        consecutive_losses=10,
        previous_stake=25.0,
        bankroll=9976.0,
        payout=0.95,
        metrics=metrics,
        soft_recovery=soft,
    )
    floor = 9976.0 * 0.0025
    cap = 9976.0 * 0.025
    assert stake == pytest.approx(floor)
    assert stake < cap
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_force_explore_reason") == "infeasible"
    assert metrics.get("recovery_explore_used_cover") is False
    assert metrics.get("recovery_infeasible") is True


def test_low_hurst_without_pending_still_forces_explore():
    metrics = {"indicators": {"hurst": 0.35}}
    stake = apply_soft_recovery_stake(
        pending_total=0.0,
        base_unit=15.0,
        consecutive_losses=0,
        previous_stake=0.0,
        bankroll=11500.0,
        payout=0.95,
        metrics=metrics,
    )
    floor = 11500.0 * 0.0025
    assert stake == pytest.approx(floor)
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_force_explore_reason") == "low_hurst"


def test_small_account_hard_floor_caps_recovery_at_five_percent():
    soft = {"max_safe_stake_cap": 4.20}
    assert max_safe_stake_cap(80.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(75.0, consecutive_losses_linear=5, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.20)
