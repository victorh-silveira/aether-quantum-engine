import pytest

from src.domain.risk.stake_sizing import (
    _resolve_stop_win_max_stake_pct,
    clamp_kelly_stake,
    compute_single_strike_kelly_base,
)


def test_resolve_stop_win_max_stake_pct_explicit_zero_disables_cap():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {"stop_win_max_stake_pct": 0}, 0.95)
    assert pct == 0.0


def test_clamp_kelly_stake_unlimited_when_max_pct_zero():
    stake = clamp_kelly_stake(10000.0, 2500.0, {"max_stake_pct": 0}, 0.70)
    assert stake == pytest.approx(2500.0, rel=1e-6)


def test_compute_single_strike_unlimited_when_kelly_cap_zero():
    boosted = compute_single_strike_kelly_base(
        50.0,
        10000.0,
        0.95,
        0.75,
        {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0},
        {
            "stop_win_kelly_enabled": True,
            "stop_win_kelly_min_conviction": 0.45,
            "stop_win_kelly_conviction_strong": 0.72,
            "stop_win_kelly_min_fraction": 0.42,
            "stop_win_kelly_max_fraction": 1.0,
            "max_stake_pct": 0,
            "stop_win_max_stake_pct": 0,
        },
        10000.0,
        0.0,
        has_active_contracts=False,
    )
    assert boosted > 50.0


def test_compute_single_strike_uses_stop_cap_when_kelly_cap_disabled():
    boosted = compute_single_strike_kelly_base(
        50.0,
        10000.0,
        0.95,
        0.75,
        {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0},
        {
            "stop_win_kelly_enabled": True,
            "stop_win_kelly_min_conviction": 0.45,
            "stop_win_kelly_conviction_strong": 0.72,
            "stop_win_kelly_min_fraction": 0.42,
            "stop_win_kelly_max_fraction": 1.0,
            "max_stake_pct": 0,
        },
        10000.0,
        0.0,
        has_active_contracts=False,
    )
    stop_cap_stake = (0.04 / 0.95) * 10000.0
    assert boosted <= stop_cap_stake + 1.0
    assert boosted > 50.0
