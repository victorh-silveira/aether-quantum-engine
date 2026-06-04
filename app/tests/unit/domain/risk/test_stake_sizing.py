import datetime
from unittest.mock import patch

import pytest

from src.domain.risk.stake_sizing import compute_single_strike_kelly_base, martingale_log_suffix, martingale_stake


def test_compute_single_strike_returns_kelly_when_not_eligible():
    kelly = compute_single_strike_kelly_base(
        50.0,
        1000.0,
        0.95,
        0.6,
        {},
        {},
        1000.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 50.0


def test_compute_single_strike_keeps_kelly_when_boost_not_greater():
    fixed = datetime.datetime(2026, 6, 2, 14, 0, 0, tzinfo=datetime.UTC)
    with patch("src.domain.risk.stake_sizing.datetime") as mock_dt:
        mock_dt.datetime.now.return_value = fixed
        mock_dt.UTC = datetime.UTC
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


def test_martingale_log_suffix():
    assert martingale_log_suffix("KELLY", 10.0, 5.0, 2.0, 0.95) == ""
    suffix = martingale_log_suffix("MARTINGALE", 100.0, 50.0, 10.0, 0.95)
    assert "MARTINGALE" in suffix
    assert "100.00" in suffix


def test_martingale_native_doubles_from_last_stake():
    cfg = {"martingale_multiplier": 2.0, "max_recovery_stake_pct": 0.10, "min_stake_pct": 0.0}
    stake = martingale_stake(
        10000.0,
        30.0,
        10.0,
        0.95,
        cfg,
        0.7,
        1.0,
        12000.0,
        consecutive_losses=2,
        last_martingale_stake=20.46,
    )
    assert stake == pytest.approx(40.92, abs=0.02)


def test_martingale_native_doubles_lost_entry_stake():
    cfg = {"martingale_multiplier": 2.0, "max_recovery_stake_pct": 0.10, "min_stake_pct": 0.0}
    stake = martingale_stake(
        10000.0,
        10.83,
        86.0,
        0.95,
        cfg,
        0.7,
        1.0,
        12000.0,
        last_martingale_stake=0.0,
        last_loss_stake=10.83,
    )
    assert stake == pytest.approx(21.66, abs=0.02)


def test_martingale_native_base_doubles_kelly_when_no_last_stake():
    cfg = {"martingale_multiplier": 2.0, "max_recovery_stake_pct": 0.10, "min_stake_pct": 0.0}
    stake = martingale_stake(
        10000.0,
        10.0,
        10.0,
        0.95,
        cfg,
        0.7,
        1.0,
        12000.0,
        consecutive_losses=1,
        last_martingale_stake=0.0,
        last_loss_stake=0.0,
    )
    assert stake == pytest.approx(20.0, abs=0.02)
