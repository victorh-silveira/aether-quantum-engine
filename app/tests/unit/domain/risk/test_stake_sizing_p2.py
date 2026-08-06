from src.domain.risk.stake_sizing import compute_single_strike_kelly_base


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
        live_metrics={"live_n": 40, "live_wr": 0.55},
    )
    assert kelly == 12.0
