from types import SimpleNamespace

import pytest

from src.application.services.execution_quality_gate_drawdown import (
    apply_dynamic_recovery_relaxation,
    recovery_drawdown_quality_limits,
    recovery_neutral_edge_zscore_waiver,
    resolve_session_stake_unit,
)
from src.application.services.execution_runtime_config import resolve_quality_gate_from_exec


def _full_pending_units() -> float:
    return float(resolve_quality_gate_from_exec(None)["recovery_relax"]["full_pending_units"])


def test_resolve_session_stake_unit_reads_risk_manager_session_base_unit():
    risk_manager = SimpleNamespace(session_base_unit=20.0)
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(20.0)


def test_resolve_session_stake_unit_falls_back_to_bankroll_fraction():
    risk_manager = SimpleNamespace(initial_bankroll=2000.0)
    assert resolve_session_stake_unit(risk_manager, {}) == pytest.approx(3.0)


def test_apply_dynamic_recovery_relaxation_noop_without_linear_or_pending():
    margin, edge, intensity = apply_dynamic_recovery_relaxation(
        0.12,
        0.04,
        linear=1,
        pending=6.75,
        session_unit=1.0,
    )
    assert margin == pytest.approx(0.12)
    assert edge == pytest.approx(0.04)
    assert intensity == pytest.approx(0.0)


def test_apply_dynamic_recovery_relaxation_scales_with_pending():
    unit = 1.0
    pending = 6.75
    expected_intensity = min(1.0, pending / (_full_pending_units() * unit))
    margin, edge, intensity = apply_dynamic_recovery_relaxation(
        0.12,
        0.04,
        linear=2,
        pending=pending,
        session_unit=unit,
        exec_cfg={
            "quality_gate": {
                "recovery_relax": {
                    "min_linear": 2,
                    "margin_floor": 0.02,
                    "edge_floor": 0.0,
                    "full_pending_units": 8.0,
                    "edge_zscore_waiver": 0.5,
                    "session_stake_unit_bankroll_pct": 0.0015,
                }
            }
        },
    )
    assert intensity == pytest.approx(expected_intensity)
    assert margin == pytest.approx(0.12 * (1.0 - expected_intensity) + 0.02 * expected_intensity)
    assert edge < 0.04
    assert edge >= 0.0


def test_recovery_drawdown_quality_limits_higher_pending_lowers_floors():
    recovery = {"min_direction_margin": 0.12, "min_payoff_edge": 0.04}
    regular = {"min_direction_margin": 0.06, "min_payoff_edge": 0.01}
    light_margin, light_edge = recovery_drawdown_quality_limits(
        recovery,
        regular,
        pending=2.0,
        session_unit=1.0,
        linear=2,
    )
    heavy_margin, heavy_edge = recovery_drawdown_quality_limits(
        recovery,
        regular,
        pending=32.0,
        session_unit=1.0,
        linear=2,
    )
    assert heavy_margin < light_margin
    assert heavy_edge < light_edge


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


def test_recovery_neutral_edge_zscore_waiver_accepts_vacuum_with_conviction():
    assert recovery_neutral_edge_zscore_waiver(0.01, 0.51, linear=2, pending=6.75) is True


def test_recovery_neutral_edge_zscore_waiver_rejects_low_z_or_out_of_band():
    assert recovery_neutral_edge_zscore_waiver(0.01, 0.50, linear=2, pending=6.75) is False
    assert recovery_neutral_edge_zscore_waiver(-0.06, 0.80, linear=2, pending=6.75) is False
    assert recovery_neutral_edge_zscore_waiver(0.05, 0.80, linear=2, pending=6.75) is False
    assert recovery_neutral_edge_zscore_waiver(0.01, 0.80, linear=1, pending=6.75) is False
    assert recovery_neutral_edge_zscore_waiver(0.01, 0.80, linear=2, pending=0.0) is False
