import math

import pytest

from src.domain.risk.martingale_sizing import (
    martingale_log_suffix,
    martingale_stake,
    martingale_stop_win_floor,
    resolve_mode_stake,
)


def test_martingale_log_suffix():
    assert martingale_log_suffix("KELLY", 10.0, 5.0, 2.0, 0.95) == ""
    suffix = martingale_log_suffix("MARTINGALE", 100.0, 50.0, 10.0, 0.95)
    assert "MARTINGALE" in suffix
    assert "100.00" in suffix


def test_martingale_covers_pending_loss_and_profit_target():
    cfg = {"min_stake_pct": 0.0, "martingale_target_fraction": 1.0}
    stake = martingale_stake(
        10000.0,
        10.83,
        86.0,
        0.95,
        cfg,
        1.0,
        last_loss_stake=10.83,
    )
    expected = (10.83 + 10.83 * 0.95) / 0.95
    assert stake == pytest.approx(math.ceil(expected * 100) / 100, abs=0.02)


def test_martingale_scales_profit_target_for_fast_cycles():
    cfg = {"min_stake_pct": 0.0, "martingale_target_fraction": 0.45}
    full = martingale_stake(10000.0, 50.0, 5.0, 0.95, {"min_stake_pct": 0.0}, 1.0, last_loss_stake=5.0)
    scaled = martingale_stake(10000.0, 50.0, 5.0, 0.95, cfg, 1.0, last_loss_stake=5.0)
    assert scaled < full


def test_martingale_scales_with_pending_loss():
    cfg = {"min_stake_pct": 0.0}
    low = martingale_stake(10000.0, 10.0, 10.0, 0.95, cfg, 1.0)
    high = martingale_stake(10000.0, 500.0, 10.0, 0.95, cfg, 1.0, last_loss_stake=50.0)
    assert high > low


def test_martingale_stop_win_floor_raises_recovery_stake():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_martingale_progress_fraction": 0.32,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
        "min_stake_pct": 0.0,
    }
    stake, recovery, mode = resolve_mode_stake(
        martingale_active=True,
        bankroll=1168.0,
        loss_to_recover=1.17,
        kelly_base=1.17,
        payout=0.95,
        kelly_config=cfg,
        stake_min=1.0,
        last_loss_stake=1.17,
        conviction=0.58,
        risk_config=risk,
        initial_bankroll=1168.0,
        total_session_profit=-1.17,
    )
    assert mode == "MARTINGALE"
    floor = martingale_stop_win_floor(
        1168.0,
        0.95,
        0.58,
        risk,
        cfg,
        1168.0,
        -1.17,
    )
    assert recovery >= floor
    assert stake > 2.5


def test_martingale_stop_win_floor_zero_when_payout_invalid():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_martingale_progress_fraction": 0.32,
    }
    floor = martingale_stop_win_floor(
        1168.0,
        0.0,
        0.58,
        risk,
        cfg,
        1168.0,
        0.0,
    )
    assert floor == 0.0


def test_martingale_stop_win_floor_zero_when_conviction_low():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_martingale_progress_fraction": 0.32,
    }
    floor = martingale_stop_win_floor(
        1168.0,
        0.95,
        0.30,
        risk,
        cfg,
        1168.0,
        0.0,
    )
    assert floor == 0.0


def test_martingale_stop_win_floor_zero_when_target_reached():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_martingale_progress_fraction": 0.32,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
    }
    floor = martingale_stop_win_floor(
        1168.0,
        0.95,
        0.58,
        risk,
        cfg,
        1168.0,
        50.0,
    )
    assert floor == 0.0


def test_martingale_recovery_without_pct_cap():
    cfg = {"min_stake_pct": 0.0}
    stake = martingale_stake(
        1300.0,
        82.0,
        10.0,
        0.95,
        cfg,
        1.0,
        last_loss_stake=50.0,
    )
    expected = (82.0 + 50.0 * 0.95) / 0.95
    assert stake == pytest.approx(expected, abs=0.02)


def test_martingale_recovery_covers_effective_loss_floor():
    cfg = {
        "min_stake_pct": 0.0,
        "martingale_target_fraction": 0.0,
        "martingale_recovery_step_fraction": 0.40,
        "martingale_max_stake_multiplier": 10.0,
    }
    stake = martingale_stake(10000.0, 100.0, 7.0, 0.95, cfg, 1.0, last_loss_stake=7.0)
    assert stake == pytest.approx(100.0 * 0.40 / 0.95, abs=0.02)


def test_martingale_recovery_step_fraction_limits_stake():
    cfg = {
        "min_stake_pct": 0.0,
        "martingale_target_fraction": 1.0,
        "martingale_recovery_step_fraction": 0.40,
        "martingale_max_stake_multiplier": 2.5,
    }
    full = martingale_stake(10000.0, 100.0, 7.0, 0.95, {"min_stake_pct": 0.0}, 1.0, last_loss_stake=7.0)
    stepped = martingale_stake(10000.0, 100.0, 7.0, 0.95, cfg, 1.0, last_loss_stake=7.0)
    assert stepped < full
    assert stepped <= 7.0 * 2.5 + 0.01


def test_martingale_limited_by_bankroll():
    cfg = {"min_stake_pct": 0.0}
    stake = martingale_stake(
        100.0,
        5000.0,
        10.0,
        0.95,
        cfg,
        1.0,
        last_loss_stake=50.0,
    )
    assert stake == pytest.approx(100.0, abs=0.02)
