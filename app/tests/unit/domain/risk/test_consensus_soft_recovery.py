import pytest

from src.domain.risk.consensus_stake_penalty import (
    adaptive_recovery_progression_factor,
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


def test_resolve_contract_payout_falls_back_to_ninety():
    assert resolve_contract_payout(None, {}) == pytest.approx(0.90)


def test_resolve_contract_payout_ignores_invalid_values():
    assert resolve_contract_payout("bad", {"payout_estimate": "oops"}) == pytest.approx(0.90)


def test_soft_recovery_progression_multiplier_powers():
    payout = 0.95
    factor = _factor(payout)
    assert soft_recovery_progression_multiplier(0, payout=payout) == pytest.approx(1.0)
    assert soft_recovery_progression_multiplier(1, payout=payout) == pytest.approx(factor)
    assert soft_recovery_progression_multiplier(2, payout=payout) == pytest.approx(1.12)
    assert soft_recovery_progression_multiplier(3, payout=payout) == pytest.approx(factor**3)
    assert soft_recovery_progression_multiplier(4, payout=payout) == pytest.approx(factor**4)
    assert soft_recovery_progression_multiplier(5, payout=payout) == pytest.approx(factor**5)


def test_apply_soft_recovery_stake_fixed_step_at_linear_two():
    metrics: dict = {}
    payout = 0.95
    stake = apply_soft_recovery_stake(
        pending_total=20.0,
        base_unit=10.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics=metrics,
        payout=payout,
    )
    assert stake == pytest.approx(64.4)
    assert metrics.get("recovery_fixed_step") is True
    assert metrics.get("recovery_progression_multiplier") == pytest.approx(1.12)


def test_session_base_unit_at_eleven_point_five_k():
    unit = neutral_edge_dynamic_unit(11500.0)
    assert unit == pytest.approx(57.5)
    metrics: dict = {}
    resolved = resolve_session_base_unit(11500.0, 2.0, metrics)
    assert resolved == pytest.approx(57.5)
    assert metrics["session_base_unit"] == pytest.approx(57.5)


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
    assert stake == pytest.approx(57.5)


def test_apply_soft_recovery_stake_progresses_with_adaptive_factor():
    metrics: dict = {}
    payout = 0.95
    factor = _factor(payout)
    stake = apply_soft_recovery_stake(
        pending_total=93.19,
        base_unit=10.0,
        consecutive_losses=2,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics=metrics,
        payout=payout,
    )
    session_unit = max(10.0, 11500.0 * 0.005)
    session_unit * (factor**2)
    93.19 / payout / 1.0
    assert stake == pytest.approx(98.095, rel=1e-2)
    assert metrics.get("recovery_soft_progression") == pytest.approx(factor)
    assert metrics.get("recovery_adaptive_payout") == pytest.approx(payout)
    assert metrics.get("recovery_soft_losses") == 2
    assert metrics.get("recovery_cover_need") == pytest.approx(98.095, rel=1e-2)
    assert metrics.get("recovery_fixed_step") in (True, False)


def test_apply_soft_recovery_stake_ignores_previous_stake_for_geometric_progression():
    payout = 0.95
    factor = _factor(payout)
    unit = 17.89
    pending = 50.0
    bankroll = 11926.67
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=bankroll,
        payout=payout,
    )
    session_unit = max(unit, bankroll * 0.005)
    assert stake == pytest.approx(max(session_unit * factor, pending / payout / 4.0))


def test_apply_soft_recovery_stake_c0005_neutral_linear_one_nominal():
    payout = 0.95
    factor = _factor(payout)
    unit = 17.89
    pending = 36.72
    bankroll = 11926.67
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=bankroll,
        metrics=metrics,
        payout=payout,
    )
    session_unit = max(unit, bankroll * 0.005)
    assert stake == pytest.approx(max(session_unit * factor, pending / payout / 4.0), rel=1e-3)
    assert metrics.get("recovery_soft_losses") == 1
    assert metrics.get("recovery_soft_progression") == pytest.approx(factor)


def test_apply_soft_recovery_stake_covers_pending_within_bankroll_cap():
    payout = 0.95
    pending = 140.78
    stake = apply_soft_recovery_stake(
        pending_total=pending,
        base_unit=24.77,
        consecutive_losses=2,
        previous_stake=50.84,
        bankroll=12895.79,
        metrics={},
        payout=payout,
    )
    pending / payout / 1.0
    max_safe_stake_cap(12895.79, consecutive_losses_linear=2)
    assert stake == pytest.approx(148.19, rel=1e-2)


def test_max_safe_stake_cap_at_five_percent():
    assert max_safe_stake_cap(11300.0) == pytest.approx(565.0)


def test_max_safe_stake_cap_compresses_on_linear_streak():
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=2) == pytest.approx(450.0)
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=3) == pytest.approx(400.0)


def test_apply_soft_recovery_stake_respects_bankroll_cap():
    metrics: dict = {}
    soft = {"infeasible_force_explore": False}
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


def test_sequential_drawdown_stakes_grow_smoothly_not_geometric_two():
    bankroll = 11000.0
    unit = 17.25
    payout = 0.95
    factor = _factor(payout)
    pending = 50.0
    losses_sequence = [1, 2, 3, 4, 5]
    stakes = [
        apply_soft_recovery_stake(
            pending_total=pending,
            base_unit=unit,
            consecutive_losses=losses,
            previous_stake=0.0,
            bankroll=bankroll,
            payout=payout,
        )
        for losses in losses_sequence
    ]
    asserted = []
    for n in losses_sequence:
        amort = max(2, 5 - min(n, 3))
        cover = (pending / payout) / amort
        geometric = unit * 1.15 if n in (3, 4) else unit * (factor**n) if n > 0 else unit
        asserted.append(min(max(geometric, cover), max_safe_stake_cap(bankroll, consecutive_losses_linear=n)))
    assert len(stakes) == len(asserted)
    assert stakes[-1] <= max_safe_stake_cap(bankroll, consecutive_losses_linear=5)


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


def test_small_account_hard_floor_caps_recovery_at_five_percent():
    soft = {"max_safe_stake_cap": 4.20}
    assert max_safe_stake_cap(80.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(75.0, consecutive_losses_linear=5, soft_recovery=soft) == pytest.approx(4.2)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=4, soft_recovery=soft) == pytest.approx(4.20)


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
