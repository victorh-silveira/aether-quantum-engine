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
    assert soft_recovery_progression_multiplier(3, payout=payout) == pytest.approx(factor**3)


def test_session_base_unit_at_eleven_point_five_k():
    unit = neutral_edge_dynamic_unit(11500.0)
    assert unit == pytest.approx(17.25)
    metrics: dict = {}
    resolved = resolve_session_base_unit(11500.0, 2.0, metrics)
    assert resolved == pytest.approx(17.25)
    assert metrics["session_base_unit"] == pytest.approx(17.25)


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
    assert stake == pytest.approx(17.25)


def test_apply_soft_recovery_stake_progresses_with_adaptive_factor():
    metrics: dict = {}
    payout = 0.95
    factor = _factor(payout)
    stake = apply_soft_recovery_stake(
        pending_total=93.19,
        base_unit=10.0,
        consecutive_losses=3,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics=metrics,
        payout=payout,
    )
    session_unit = max(10.0, 11500.0 * 0.0015)
    assert stake == pytest.approx(session_unit * (factor**3))
    assert metrics.get("recovery_soft_progression") == pytest.approx(factor)
    assert metrics.get("recovery_adaptive_payout") == pytest.approx(payout)
    assert metrics.get("recovery_soft_losses") == 3


def test_apply_soft_recovery_stake_ignores_previous_stake_for_geometric_progression():
    payout = 0.95
    factor = _factor(payout)
    unit = 17.89
    stake = apply_soft_recovery_stake(
        pending_total=50.0,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=11926.67,
        payout=payout,
    )
    assert stake == pytest.approx(unit * factor)


def test_apply_soft_recovery_stake_c0005_neutral_linear_one_nominal():
    payout = 0.95
    factor = _factor(payout)
    unit = 17.89
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=36.72,
        base_unit=unit,
        consecutive_losses=1,
        previous_stake=36.72,
        bankroll=11926.67,
        metrics=metrics,
        payout=payout,
    )
    assert stake == pytest.approx(unit * factor, rel=1e-3)
    assert metrics.get("recovery_soft_losses") == 1
    assert metrics.get("recovery_soft_progression") == pytest.approx(factor)


def test_max_safe_stake_cap_at_three_point_five_percent():
    assert max_safe_stake_cap(11300.0) == pytest.approx(395.50)


def test_max_safe_stake_cap_compresses_on_linear_streak():
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=2) == pytest.approx(250.0)
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=3) == pytest.approx(200.0)


def test_apply_soft_recovery_stake_respects_bankroll_cap():
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=5000.0,
        base_unit=200.0,
        consecutive_losses=8,
        previous_stake=0.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.70,
    )
    assert stake == pytest.approx(max_safe_stake_cap(10000.0, consecutive_losses_linear=8))


def test_sequential_drawdown_stakes_grow_smoothly_not_geometric_two():
    bankroll = 11000.0
    unit = 17.25
    payout = 0.95
    factor = _factor(payout)
    losses_sequence = [1, 2, 3, 4, 5]
    stakes = [
        apply_soft_recovery_stake(
            pending_total=50.0,
            base_unit=unit,
            consecutive_losses=losses,
            previous_stake=0.0,
            bankroll=bankroll,
            payout=payout,
        )
        for losses in losses_sequence
    ]
    assert stakes[2] < stakes[0] * 8
    assert stakes[-1] == pytest.approx(
        min(unit * (factor**5), max_safe_stake_cap(bankroll, consecutive_losses_linear=5))
    )


def test_adaptive_recovery_factor_at_payout_ninety_five():
    assert adaptive_recovery_progression_factor(0.95) == pytest.approx(2.0526315789)


def test_adaptive_recovery_factor_expands_at_payout_seventy_five():
    assert adaptive_recovery_progression_factor(0.75) == pytest.approx(2.3333333333)


def test_adaptive_recovery_factor_expands_at_payout_seventy():
    assert adaptive_recovery_progression_factor(0.70) == pytest.approx(2.4285714286)


def test_adaptive_recovery_factor_caps_at_two_fifty():
    assert adaptive_recovery_progression_factor(0.40) == pytest.approx(2.50)


def test_soft_recovery_progression_multiplier_scales_with_payout_stress():
    high = soft_recovery_progression_multiplier(3, payout=0.95)
    low = soft_recovery_progression_multiplier(3, payout=0.70)
    assert low > high


def test_apply_soft_recovery_stake_expands_when_payout_degrades():
    base_kwargs = {
        "pending_total": 80.0,
        "base_unit": 15.0,
        "consecutive_losses": 3,
        "previous_stake": 0.0,
        "bankroll": 11500.0,
    }
    stake_hi = apply_soft_recovery_stake(**base_kwargs, payout=0.95, metrics={})
    stake_lo = apply_soft_recovery_stake(**base_kwargs, payout=0.70, metrics={})
    assert stake_lo > stake_hi
