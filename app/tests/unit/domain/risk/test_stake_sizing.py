import math

import pytest

from src.domain.risk.stake_sizing import (
    _resolve_stop_win_max_stake_pct,
    compute_single_strike_kelly_base,
    conviction_stop_win_weight,
    martingale_log_suffix,
    martingale_stake,
    martingale_stop_win_floor,
    resolve_mode_stake,
)


def test_compute_single_strike_returns_kelly_when_conviction_low():
    kelly = compute_single_strike_kelly_base(
        50.0,
        1000.0,
        0.95,
        0.40,
        {"large_account_stop_win_pct": 4.0},
        {},
        1000.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 50.0


def test_compute_single_strike_keeps_kelly_when_boost_not_greater():
    kelly = compute_single_strike_kelly_base(
        5000.0,
        10000.0,
        0.95,
        0.8,
        {"large_account_stop_win_pct": 10.0, "small_account_threshold": 0.0},
        {"max_stake_pct": 0.01},
        10000.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 5000.0


def test_compute_single_strike_targets_stop_win_pct():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_conviction_strong": 0.72,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
    }
    weak = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.46,
        risk,
        cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    weight = conviction_stop_win_weight(0.46, cfg)
    assert weak == pytest.approx((46.72 / 0.95) * weight, abs=0.5)
    strong = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.75,
        risk,
        cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert strong == pytest.approx(46.72 / 0.95, abs=0.5)


def test_compute_single_strike_cycles_target_reduces_stake():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    base_cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_conviction_strong": 0.75,
        "stop_win_kelly_min_fraction": 0.12,
        "stop_win_kelly_max_fraction": 0.38,
        "stop_win_kelly_cycles_target": 1.0,
    }
    full = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        base_cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    damped_cfg = {**base_cfg, "stop_win_kelly_cycles_target": 2.75}
    damped = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        damped_cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert damped < full
    assert damped == pytest.approx(full / 2.75, abs=0.5)


def test_compute_single_strike_disabled_when_flag_off():
    kelly = compute_single_strike_kelly_base(
        12.0,
        1168.0,
        0.95,
        0.60,
        {"large_account_stop_win_pct": 4.0},
        {"stop_win_kelly_enabled": False},
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 12.0


def test_resolve_stop_win_max_stake_pct_from_stop_win():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.95)
    assert pct == pytest.approx(0.04 / 0.95, rel=1e-6)


def test_resolve_stop_win_max_stake_pct_explicit_override():
    pct = _resolve_stop_win_max_stake_pct({}, {"stop_win_max_stake_pct": 0.03}, 0.95)
    assert pct == 0.03


def test_resolve_stop_win_max_stake_pct_without_payout():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.0)
    assert pct == pytest.approx(0.04, rel=1e-6)


def test_martingale_log_suffix():
    assert martingale_log_suffix("KELLY", 10.0, 5.0, 2.0, 0.95) == ""
    suffix = martingale_log_suffix("MARTINGALE", 100.0, 50.0, 10.0, 0.95)
    assert "MARTINGALE" in suffix
    assert "100.00" in suffix


def test_martingale_covers_pending_loss_and_profit_target():
    cfg = {"min_stake_pct": 0.0}
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
