import pytest

from src.domain.risk.consensus_stake_penalty import (
    apply_soft_recovery_stake,
    max_safe_stake_cap,
    neutral_edge_dynamic_unit,
    resolve_contract_payout,
    resolve_session_base_unit,
    soft_recovery_progression_multiplier,
)


def _factor(payout: float) -> float:
    return min(1.0 + (1.0 / payout), 2.50)


def test_resolve_contract_payout_uses_api_value():
    assert resolve_contract_payout(0.95, {"payout_estimate": 0.90}) == pytest.approx(0.95)


def test_resolve_contract_payout_falls_back_to_ssot():
    assert resolve_contract_payout(None, {}) == pytest.approx(0.85)


def test_resolve_contract_payout_ignores_invalid_values():
    assert resolve_contract_payout("bad", {"payout_estimate": "oops"}) == pytest.approx(0.85)


def test_soft_recovery_progression_multiplier_powers():
    payout = 0.95
    factor = _factor(payout)
    assert soft_recovery_progression_multiplier(0, payout=payout) == pytest.approx(1.0)
    assert soft_recovery_progression_multiplier(1, payout=payout) == pytest.approx(factor)
    assert soft_recovery_progression_multiplier(2, payout=payout) == pytest.approx(1.12)
    assert soft_recovery_progression_multiplier(3, payout=payout) == pytest.approx(1.12)
    assert soft_recovery_progression_multiplier(4, payout=payout) == pytest.approx(1.12)
    assert soft_recovery_progression_multiplier(5, payout=payout) == pytest.approx(factor**5)


def test_apply_soft_recovery_stake_amort_uses_cover_not_geometric():
    metrics: dict = {}
    payout = 0.95
    pending = 20.0
    soft = {
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "cover_enabled": True,
        "cover_multiple": 1.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=10.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics=metrics,
        payout=payout,
        soft_recovery=soft,
    )
    cover = pending / payout / 3.0
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_amort_cycles") == 3
    assert metrics.get("recovery_fixed_step") is True
    assert metrics.get("recovery_progression_multiplier") == pytest.approx(1.12)
    assert stake < neutral_edge_dynamic_unit(11500.0) * 1.12


def test_apply_soft_recovery_cover_multiple_doubles_need():
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=72.0,
        base_unit=10.0,
        consecutive_losses=1,
        previous_stake=0.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.72,
        soft_recovery={
            "amort_cycles_min": 1,
            "amort_cycles_max": 1,
            "cover_enabled": True,
            "cover_multiple": 2.0,
            "max_safe_stake_pct": 0.05,
            "infeasible_force_explore": True,
            "material_pending_min": 0.25,
        },
    )
    cover = 72.0 / 0.72 * 2.0
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_cover_multiple") == pytest.approx(2.0)
    assert metrics.get("recovery_cover_need") == pytest.approx(cover)


def test_session_base_unit_at_eleven_point_five_k():
    unit = neutral_edge_dynamic_unit(11500.0)
    assert unit == pytest.approx(115.0)
    metrics: dict = {}
    resolved = resolve_session_base_unit(11500.0, 2.0, metrics)
    assert resolved == pytest.approx(115.0)
    assert metrics["session_base_unit"] == pytest.approx(115.0)


def test_apply_soft_recovery_stake_without_pending_returns_session_unit():
    stake = apply_soft_recovery_stake(
        pending_total=0.0,
        base_unit=2.0,
        consecutive_losses=0,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics={},
        payout=0.95,
    )
    assert stake == pytest.approx(115.0)


def test_apply_soft_recovery_stake_full_cover_amort_one_no_geometric():
    metrics: dict = {}
    payout = 0.82
    pending = 4.88
    soft = {
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_enabled": True,
        "cover_multiple": 2.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=25.0,
        consecutive_losses=1,
        previous_stake=25.0,
        bankroll=9750.0,
        metrics=metrics,
        payout=payout,
        soft_recovery=soft,
    )
    cover = pending / payout * 2.0
    assert stake == pytest.approx(cover)
    assert metrics.get("recovery_amort_cycles") == 1
    assert stake < 20.0


def test_apply_soft_recovery_stake_cover_not_damped_by_negative_session_pnl():
    metrics: dict = {}
    payout = 0.82
    pending = 38.0
    cover = pending / payout * 2.0
    soft = {
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_enabled": True,
        "cover_multiple": 2.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=0.53,
        consecutive_losses=1,
        previous_stake=38.0,
        bankroll=9462.62,
        metrics=metrics,
        payout=payout,
        soft_recovery=soft,
        session_pnl=-38.0,
        target_win=285.01,
    )
    assert stake == pytest.approx(cover)
    assert stake > 80.0


def test_apply_soft_recovery_stake_ignores_previous_stake_for_geometric_progression():
    payout = 0.95
    unit = 17.89
    pending = 50.0
    bankroll = 11926.67
    soft = {
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_enabled": True,
        "cover_multiple": 2.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=bankroll,
        payout=payout,
        soft_recovery=soft,
    )
    assert stake == pytest.approx(pending / payout * 2.0)


def test_apply_soft_recovery_stake_c0005_neutral_linear_one_nominal():
    payout = 0.95
    unit = 17.89
    pending = 36.72
    bankroll = 11926.67
    metrics: dict = {}
    soft = {
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_enabled": True,
        "cover_multiple": 2.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=bankroll,
        metrics=metrics,
        payout=payout,
        soft_recovery=soft,
    )
    assert stake == pytest.approx(pending / payout * 2.0, rel=1e-3)
    assert metrics.get("recovery_soft_losses") == 1


def test_apply_soft_recovery_stake_covers_pending_within_bankroll_cap():
    payout = 0.95
    pending = 140.78
    soft = {
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_enabled": True,
        "cover_multiple": 2.0,
        "material_pending_min": 0.25,
        "infeasible_force_explore": True,
        "max_safe_stake_pct": 0.05,
    }
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=24.77,
        consecutive_losses=2,
        previous_stake=50.84,
        bankroll=12895.79,
        metrics={},
        payout=payout,
        soft_recovery=soft,
    )
    cover = pending / payout * 2.0
    max_safe_stake_cap(12895.79, consecutive_losses_linear=2, soft_recovery=soft)
    assert stake == pytest.approx(cover, rel=1e-2)


def test_max_safe_stake_cap_at_five_percent():
    assert max_safe_stake_cap(11300.0) == pytest.approx(395.5)


def test_max_safe_stake_cap_compresses_on_linear_streak():
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=2) == pytest.approx(300.0)
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=3) == pytest.approx(250.0)


def test_apply_soft_recovery_stake_respects_bankroll_cap():
    metrics: dict = {}
    soft = {"infeasible_force_explore": False, "cover_enabled": True}
    stake = apply_soft_recovery_stake(
        pending_total=5000.0,
        base_unit=200.0,
        consecutive_losses=8,
        previous_stake=0.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.70,
        soft_recovery=soft,
    )
    assert stake == pytest.approx(max_safe_stake_cap(10000.0, consecutive_losses_linear=8, soft_recovery=soft))
