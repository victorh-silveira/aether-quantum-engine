"""Testes do monitor Rich CLI."""

import math

import pytest

from scripts.monitor.monitor_state import DashboardState, resolve_session_financials


def test_resolve_session_financials_compounding_goal():
    state = DashboardState(
        balance=9101.04,
        session_start_balance=9101.04,
        session_target_win=236.62,
        session_profit=0.0,
        compounding_rate=0.026,
    )
    fin = resolve_session_financials(state)
    assert fin.target_win == 236.62
    assert fin.target_balance == pytest.approx(9337.66, rel=1e-4)
    assert fin.remaining == 236.62
    assert "(2.6% SES. ATIVA)" in fin.goal_label


def test_resolve_session_financials_fallback_from_rate():
    state = DashboardState(
        balance=9101.04,
        session_start_balance=9101.04,
        session_target_win=0.0,
        session_profit=10.0,
        compounding_rate=0.026,
        compounding_enabled=True,
    )
    fin = resolve_session_financials(state)
    expected_target = math.floor(9101.04 * 0.026 * 100) / 100
    assert fin.target_win == expected_target
    assert fin.remaining == pytest.approx(expected_target - 10.0, rel=1e-4)


def test_resolve_session_financials_progress_with_profit():
    state = DashboardState(
        session_start_balance=1000.0,
        session_target_win=10.0,
        session_profit=5.0,
    )
    fin = resolve_session_financials(state)
    assert fin.progress_pct == 50.0
    assert fin.remaining == 5.0
