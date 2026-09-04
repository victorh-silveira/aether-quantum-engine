from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _soft_path_rm(cfg: dict, **overrides):
    rm = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        kelly_config=cfg["kelly"],
        soft_recovery_config=cfg["soft_recovery"],
        dlambert_config=cfg["dlambert"],
        risk_params=dict(cfg["params"]),
        pending_loss={},
        consecutive_losses_linear=0,
        last_loss_stake=0.0,
        dlambert_unit=0.0,
        logger=MagicMock(),
        initial_bankroll=100.0,
        total_session_profit=0.0,
        _last_stake_audit=None,
        effective_win_rate=MagicMock(return_value=0.55),
        _recovery_allowed=MagicMock(return_value=True),
    )
    for key, value in overrides.items():
        setattr(rm, key, value)
    return rm


def test_calculate_stake_recover_cover_disabled_uses_kelly_floor(kelly_config):
    soft = {**kelly_config["soft_recovery"], "cover_enabled": False, "max_safe_stake_pct": 0.05}
    rm = _soft_path_rm(
        kelly_config,
        soft_recovery_config=soft,
        consecutive_losses_linear=2,
        last_loss_stake=2.0,
        pending_loss={"R_10": 1.5},
    )
    stake = calculate_stake_for_manager(
        rm,
        bankroll=100.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 1, "dl_metrics": {"execute": True}},
    )
    floor = 100.0 * float(rm.kelly_config["neutral_bankroll_pct"])
    assert stake == pytest.approx(floor)
    assert stake <= 4.20 + 1e-9
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    assert rm._last_stake_audit["stake_regime"] == "EXPLORE"


def test_calculate_stake_explore_uses_kelly(kelly_config):
    cfg = dict(kelly_config)
    cfg["kelly"] = {**kelly_config["kelly"], "fraction": 0.25, "max_stake_pct": 0.05}
    rm = _soft_path_rm(cfg)
    rm.kelly_config = cfg["kelly"]
    stake = calculate_stake_for_manager(
        rm,
        bankroll=10000.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 7, "dl_metrics": {"execute": True, "live_wr": 0.55, "live_n": 20}},
    )
    assert stake > 1.0
    assert stake >= 10000.0 * float(rm.kelly_config["neutral_bankroll_pct"]) - 1e-6
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    assert rm._last_stake_audit["stake_regime"] == "EXPLORE"


def test_calculate_stake_recover_cover_disabled_large_pending(kelly_config):
    soft = {**kelly_config["soft_recovery"], "cover_enabled": False, "max_safe_stake_pct": 0.035}
    rm = _soft_path_rm(
        kelly_config,
        soft_recovery_config=soft,
        consecutive_losses_linear=1,
        last_loss_stake=38.46,
        pending_loss={"R_10": 38.46},
        initial_bankroll=12000.0,
    )
    stake = calculate_stake_for_manager(
        rm,
        bankroll=12000.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 4, "dl_metrics": {"execute": True}},
    )
    floor = 12000.0 * float(rm.kelly_config["neutral_bankroll_pct"])
    cover_legacy = 38.46 / 0.72 * 1.50
    assert stake == pytest.approx(floor)
    assert stake != pytest.approx(cover_legacy)
    assert stake <= 12000.0 * 0.035 + 1e-9
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    assert rm._last_stake_audit["stake_regime"] == "EXPLORE"


def test_calculate_stake_large_bankroll_soft_cap_uses_pct_not_abs(kelly_config):
    soft = {**kelly_config["soft_recovery"], "cover_enabled": False, "max_safe_stake_pct": 0.035}
    cap = max_safe_stake_cap(12000.0, consecutive_losses_linear=1, soft_recovery=soft)
    assert cap == pytest.approx(420.0)
    assert cap > 4.20
    rm = _soft_path_rm(
        kelly_config,
        soft_recovery_config=soft,
        consecutive_losses_linear=1,
        last_loss_stake=20.0,
        pending_loss={"R_10": 20.0},
        initial_bankroll=12000.0,
    )
    stake = calculate_stake_for_manager(
        rm,
        bankroll=12000.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 9, "dl_metrics": {"execute": True}},
    )
    floor = 12000.0 * float(rm.kelly_config["neutral_bankroll_pct"])
    assert stake == pytest.approx(floor)
    assert stake <= cap + 1e-9
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    assert rm._last_stake_audit["cap"] == pytest.approx(cap)


def test_calculate_stake_cover_disabled_preserves_soft_size_with_pend(kelly_config):
    soft = {**kelly_config["soft_recovery"], "cover_enabled": False, "max_safe_stake_pct": 0.035}
    kelly = {
        **kelly_config["kelly"],
        "stop_win_kelly_enabled": True,
        "soft_size_min_stake_pct": 0.025,
        "soft_size_max_stake_pct": 0.025,
        "soft_size_min_edge": 0.015,
        "neutral_bankroll_pct": 0.01,
        "min_stake_pct": 0.01,
    }
    rm = _soft_path_rm(
        kelly_config,
        kelly_config=kelly,
        soft_recovery_config=soft,
        consecutive_losses_linear=1,
        last_loss_stake=237.0,
        pending_loss={"R_10": 237.0},
        initial_bankroll=9500.0,
    )
    bankroll = 9267.0
    stake = calculate_stake_for_manager(
        rm,
        bankroll=bankroll,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=True,
        kwargs={
            "cycle_id": 2,
            "dl_metrics": {
                "execute": True,
                "gate_verdict": "SOFT_SIZE",
                "anti_loss_soft": True,
                "neg_edge_tcn_cal_edge": 0.1146,
                "live_n": 1,
                "live_wr": 0.0,
            },
        },
    )
    soft_floor = bankroll * 0.025
    one_pct = bankroll * 0.01
    cap = max_safe_stake_cap(bankroll, consecutive_losses_linear=1, soft_recovery=soft)
    assert stake == pytest.approx(min(soft_floor, cap), abs=0.02)
    assert stake > one_pct + 1.0
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"


def test_calculate_stake_recover_cover_enabled_uses_dal(kelly_config):
    soft = {
        **kelly_config["soft_recovery"],
        "cover_enabled": True,
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_multiple": 1.50,
        "max_safe_stake_pct": 0.05,
        "max_safe_stake_cap": 4.20,
    }
    rm = _soft_path_rm(
        kelly_config,
        soft_recovery_config=soft,
        consecutive_losses_linear=2,
        last_loss_stake=2.0,
        pending_loss={"R_10": 1.5},
    )
    stake = calculate_stake_for_manager(
        rm,
        bankroll=100.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 1, "dl_metrics": {"execute": True}},
    )
    assert stake != pytest.approx(4.0)
    assert stake <= 4.20 + 1e-9
    assert stake > 0.0
    assert rm._last_stake_audit["mode_tag"] == "RECOVER_DAL_L2"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"


def test_calculate_stake_explore_after_reset_uses_kelly(kelly_config):
    rm = _soft_path_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        bankroll=5000.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 3, "dl_metrics": {"execute": True, "live_wr": 0.55, "live_n": 12}},
    )
    assert stake >= 5000.0 * float(rm.kelly_config["neutral_bankroll_pct"]) - 1e-6
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
