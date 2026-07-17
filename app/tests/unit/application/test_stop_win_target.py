import pytest

from src.domain.risk.stop_win_target import (
    SessionTargets,
    StopWinManager,
    resolve_max_stake_pct,
    resolve_session_start_balance,
    resolve_stop_win_target,
)


def test_resolve_stop_win_small_account_fixed():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "large_account_stop_win_pct": 50.0,
        "params": {"compounding_enabled": False},
    }
    assert resolve_stop_win_target(rm, 50.0) == pytest.approx(10.0)


def test_resolve_stop_win_large_account_pct():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "large_account_stop_win_pct": 1.0,
        "params": {"compounding_enabled": False},
    }
    assert resolve_stop_win_target(rm, 1000.0) == pytest.approx(10.0)


def test_session_targets_compounding_rate():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "params": {"compounding_enabled": True, "compounding_rate_daily": 0.026},
    }
    mgr = StopWinManager(rm)
    targets = mgr.calculate_session_targets(10000.0)
    assert isinstance(targets, SessionTargets)
    assert targets.target_win == pytest.approx(260.0)
    assert targets.session_start_balance == pytest.approx(10000.0)


def test_session_targets_micro_bankroll_fixed_even_with_compounding():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "params": {"compounding_enabled": True, "compounding_rate_daily": 0.026},
    }
    mgr = StopWinManager(rm)
    assert mgr.is_small_account(99.99) is True
    assert mgr.is_small_account(100.0) is False
    assert mgr.calculate_session_targets(50.0).target_win == pytest.approx(10.0)
    assert mgr.calculate_session_targets(99.99).target_win == pytest.approx(10.0)
    assert mgr.calculate_session_targets(100.0).target_win == pytest.approx(2.60)
    assert mgr.resolve_target(75.0) == pytest.approx(10.0)


def test_persisted_target_idempotent_for_live_session():
    rm = {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.026}}
    assert resolve_stop_win_target(rm, 5000.0, persisted_target=75.0) == pytest.approx(75.0)
    assert resolve_stop_win_target(rm, 9999.0, persisted_target=75.0) == pytest.approx(75.0)


def test_resolve_session_start_balance_prefers_settings_override():
    rm = {"params": {"compounding_enabled": True, "session_start_balance": 2500.0}}
    assert resolve_session_start_balance(9000.0, rm) == pytest.approx(2500.0)
    assert resolve_session_start_balance(9000.0, {"params": {}}) == pytest.approx(9000.0)


def test_stop_win_manager_resolve_target_compounding_without_persisted():
    rm = {"params": {"compounding_enabled": True, "compounding_rate_daily": 0.026}}
    mgr = StopWinManager(rm)
    assert mgr.resolve_target(2500.0) == pytest.approx(65.0)


def test_resolve_max_stake_pct_high_conviction():
    cfg = {
        "max_stake_pct": 0.02,
        "max_stake_pct_high_conviction": 0.04,
        "high_conviction_stake_threshold": 0.85,
    }
    assert resolve_max_stake_pct(cfg, 0.70) == 0.02
    assert resolve_max_stake_pct(cfg, 0.90) == 0.04
