from unittest.mock import MagicMock

from src.domain.risk.kelly_p_align import calculate_kelly_fraction
from src.domain.risk.risk_stake_calc_helpers import cap_final_stake
from src.domain.risk.stake_sizing import (
    clamp_kelly_stake,
    compute_single_strike_kelly_base,
    enforce_min_stake_pct,
)


def test_compute_single_strike_disabled_when_flag_off():
    """Garante retorno inalterado quando stop_win_kelly_enabled for falso."""
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


def test_clamp_kelly_stake_with_max_stake_cap():
    """Verifica que clamp_kelly_stake respeita o teto absoluto max_stake."""
    cfg = {"min_stake_pct": 0.01, "max_stake_pct": 0.05, "max_stake": 15.0}
    clamped = clamp_kelly_stake(bankroll=10000.0, raw_stake=100.0, kelly_config=cfg, conviction=0.60)
    assert clamped == 15.0


def test_enforce_min_stake_pct_with_max_stake_cap():
    """Verifica que enforce_min_stake_pct nao ultrapassa max_stake absoluto."""
    cfg = {"min_stake_pct": 0.02, "max_stake": 15.0}
    metrics: dict = {}
    lifted = enforce_min_stake_pct(5.0, 10000.0, cfg, safe_cap=50.0, metrics=metrics)
    assert lifted == 15.0
    assert metrics.get("min_stake_pct_floor_applied") is True
    assert metrics.get("min_stake_pct_floor") == 15.0

    cfg_no_min = {"min_stake_pct": 0.0, "max_stake": 15.0}
    assert enforce_min_stake_pct(30.0, 10000.0, cfg_no_min) == 15.0
    assert enforce_min_stake_pct(0.0, 10000.0, cfg) == 0.0


def test_calculate_kelly_fraction_with_fractional_multiplier():
    """Verifica aplicacao de kelly_fraction conservador (Quarter-Kelly)."""
    rm = MagicMock()
    rm.risk_params = {"payout_estimate": 1.44}
    rm.kelly_config = {"kelly_fraction": 0.25, "kelly_p_floor": 0.55}
    rm.effective_win_rate.return_value = 0.60
    metrics: dict = {}
    f_star, b, p = calculate_kelly_fraction(rm, "1HZ75V", 0.60, metrics)
    raw_f = (1.44 * 0.60 - 0.40) / 1.44
    expected_f = raw_f * 0.25
    assert b == 1.44
    assert p == 0.60
    assert abs(f_star - expected_f) < 1e-6
    assert metrics.get("kelly_fraction_multiplier") == 0.25
    assert abs(metrics.get("kelly_fraction_raw") - raw_f) < 1e-6


def test_cap_final_stake_with_max_stake_cap():
    """Verifica teto max_stake absoluto em cap_final_stake."""
    rm = MagicMock()
    rm.soft_recovery_config = None
    rm.kelly_config = {"max_stake_pct": 0.05, "max_stake": 15.0}
    capped, safe_cap = cap_final_stake(
        100.0,
        bankroll=10000.0,
        conviction=0.60,
        recovery_stress=False,
        linear_losses=0,
        rm=rm,
    )
    assert capped == 15.0
    assert safe_cap > 0.0
