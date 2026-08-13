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


def test_calculate_stake_recover_uses_soft_not_double(kelly_config):
    rm = _soft_path_rm(kelly_config, consecutive_losses_linear=2, last_loss_stake=2.0, pending_loss={"R_10": 1.5})
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
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    assert rm._last_stake_audit["stake_regime"] == "EXPLORE"


def test_calculate_stake_recover_uses_soft_from_kelly_loss(kelly_config):
    rm = _soft_path_rm(
        kelly_config,
        consecutive_losses_linear=1,
        last_loss_stake=38.46,
        pending_loss={"R_10": 38.46},
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
    assert stake != pytest.approx(76.92)
    assert stake <= 12000.0 * 0.035 + 1e-9
    assert rm._last_stake_audit["mode_tag"] == "RECOVER_DAL_L1"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"


def test_calculate_stake_large_bankroll_soft_cap_uses_pct_not_abs(kelly_config):
    soft = {**kelly_config["soft_recovery"], "max_safe_stake_pct": 0.035}
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
    assert stake > 4.20
    assert stake <= cap + 1e-9
    assert rm._last_stake_audit["mode_tag"] == "RECOVER_DAL_L1"
    assert rm._last_stake_audit["cap"] == pytest.approx(cap)


def test_calculate_stake_explore_after_reset_uses_kelly(kelly_config):
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
        kwargs={"cycle_id": 7, "dl_metrics": {"execute": True, "live_wr": 0.52, "live_n": 10}},
    )
    assert stake > 1.0
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"
    rm.logger.debug.assert_called()
