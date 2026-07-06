import pytest

from src.domain.risk.consensus_stake_penalty import (
    apply_soft_recovery_stake,
    max_safe_stake_cap,
    neutral_edge_dynamic_unit,
    resolve_contract_payout,
    resolve_session_base_unit,
    soft_recovery_progression_multiplier,
)


def test_resolve_contract_payout_uses_api_value():
    assert resolve_contract_payout(0.95, {"payout_estimate": 0.90}) == pytest.approx(0.95)


def test_resolve_contract_payout_falls_back_to_ninety():
    assert resolve_contract_payout(None, {}) == pytest.approx(0.90)


def test_resolve_contract_payout_ignores_invalid_values():
    assert resolve_contract_payout("bad", {"payout_estimate": "oops"}) == pytest.approx(0.90)


def test_soft_recovery_progression_multiplier_powers():
    assert soft_recovery_progression_multiplier(0) == pytest.approx(1.0)
    assert soft_recovery_progression_multiplier(3) == pytest.approx(1.65**3)


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
    )
    assert stake == pytest.approx(17.25)


def test_apply_soft_recovery_stake_progresses_at_one_sixty_five():
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=93.19,
        base_unit=10.0,
        consecutive_losses=3,
        previous_stake=0.0,
        bankroll=11500.0,
        metrics=metrics,
    )
    session_unit = max(10.0, 11500.0 * 0.0015)
    assert stake == pytest.approx(session_unit * (1.65**3))
    assert metrics.get("recovery_soft_progression") == pytest.approx(1.65)
    assert metrics.get("recovery_soft_losses") == 3


def test_apply_soft_recovery_stake_uses_previous_stake_single_step():
    stake = apply_soft_recovery_stake(
        pending_total=50.0,
        base_unit=10.0,
        consecutive_losses=2,
        previous_stake=40.0,
        bankroll=11500.0,
    )
    assert stake == pytest.approx(40.0 * 1.65)


def test_max_safe_stake_cap_at_three_point_five_percent():
    assert max_safe_stake_cap(11300.0) == pytest.approx(395.50)


def test_apply_soft_recovery_stake_respects_bankroll_cap():
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=5000.0,
        base_unit=200.0,
        consecutive_losses=8,
        previous_stake=0.0,
        bankroll=10000.0,
        metrics=metrics,
    )
    assert stake == pytest.approx(max_safe_stake_cap(10000.0))


def test_sequential_drawdown_stakes_grow_smoothly_not_geometric_two():
    bankroll = 11000.0
    unit = 17.25
    losses_sequence = [1, 2, 3, 4, 5]
    stakes = [
        apply_soft_recovery_stake(
            pending_total=50.0,
            base_unit=unit,
            consecutive_losses=losses,
            previous_stake=0.0,
            bankroll=bankroll,
        )
        for losses in losses_sequence
    ]
    assert stakes[2] < stakes[0] * 8
    assert stakes[-1] == pytest.approx(min(unit * (1.65**5), max_safe_stake_cap(bankroll)))
