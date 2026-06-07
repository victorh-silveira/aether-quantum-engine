import pytest

from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


def test_resolve_stop_win_small_account_fixed():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "large_account_stop_win_pct": 50.0,
    }
    assert resolve_stop_win_target(rm, 50.0) == pytest.approx(10.0)
    assert resolve_stop_win_target(rm, 99.99) == pytest.approx(10.0)


def test_resolve_stop_win_large_account_pct():
    rm = {
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
        "large_account_stop_win_pct": 15.0,
    }
    assert resolve_stop_win_target(rm, 100.0) == pytest.approx(15.0)
    assert resolve_stop_win_target(rm, 1000.0) == pytest.approx(150.0)


def test_resolve_stop_win_pct_clamped():
    rm = {"small_account_threshold": 100.0, "large_account_stop_win_pct": 150.0}
    assert resolve_stop_win_target(rm, 200.0) == pytest.approx(200.0)


def test_resolve_max_stake_pct_recovery():
    cfg = {
        "full_recovery_martingale": True,
        "max_recovery_bankroll_pct": 0.50,
        "max_recovery_stake_pct": 0.20,
    }
    assert resolve_max_stake_pct(cfg, 0.50, is_recovery=True) == 0.50

    cfg_no_full = {
        "full_recovery_martingale": False,
        "max_recovery_stake_pct": 0.25,
    }
    assert resolve_max_stake_pct(cfg_no_full, 0.50, is_recovery=True) == 0.25
