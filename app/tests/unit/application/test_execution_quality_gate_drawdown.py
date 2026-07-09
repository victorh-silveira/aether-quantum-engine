from types import SimpleNamespace

import pytest

from src.application.services.execution_quality_gate_drawdown import (
    recovery_drawdown_quality_limits,
    resolve_session_stake_unit,
)


def test_resolve_session_stake_unit_reads_risk_manager_session_base_unit():
    risk_manager = SimpleNamespace(session_base_unit=20.0)
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(20.0)


def test_resolve_session_stake_unit_falls_back_to_bankroll_fraction():
    risk_manager = SimpleNamespace(initial_bankroll=2000.0)
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(3.0)


def test_recovery_drawdown_quality_limits_above_unit_uses_recovery():
    recovery = {"min_direction_margin": 0.12, "min_payoff_edge": 0.04}
    regular = {"min_direction_margin": 0.06, "min_payoff_edge": 0.01}
    margin, edge = recovery_drawdown_quality_limits(
        recovery,
        regular,
        pending=32.0,
        session_unit=16.0,
    )
    assert margin == pytest.approx(0.12)
    assert edge == pytest.approx(0.04)


def test_resolve_session_stake_unit_reads_configured_unit():
    exec_cfg = {"quality_gate": {"session_base_unit": 25.0}}
    assert resolve_session_stake_unit(None, exec_cfg) == pytest.approx(25.0)


def test_resolve_session_stake_unit_without_risk_manager():
    assert resolve_session_stake_unit(None, {}) == pytest.approx(1.0)


def test_resolve_session_stake_unit_reads_dlambert_unit():
    risk_manager = SimpleNamespace(dlambert_unit=18.0, session_base_unit=0.0, initial_bankroll=0.0)
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(18.0)


def test_resolve_session_stake_unit_reads_kelly_base():
    risk_manager = SimpleNamespace(
        dlambert_unit=0.0,
        session_base_unit=0.0,
        initial_bankroll=1000.0,
        kelly_config={"base_stake": 12.0},
    )
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(12.0)


def test_recovery_drawdown_quality_limits_below_unit_uses_interpolated():
    recovery = {"min_direction_margin": 0.12, "min_payoff_edge": 0.04}
    regular = {"min_direction_margin": 0.06, "min_payoff_edge": 0.01}
    margin, edge = recovery_drawdown_quality_limits(
        recovery,
        regular,
        pending=8.0,
        session_unit=16.0,
    )
    assert margin == pytest.approx(0.09)
    assert edge == pytest.approx(0.01)
