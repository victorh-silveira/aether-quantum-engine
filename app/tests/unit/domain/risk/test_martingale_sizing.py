from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.domain.risk.martingale_sizing import (
    calculate_martingale_stake_for_manager,
    martingale_enabled,
    resolve_martingale_config,
    resolve_martingale_stake,
)
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.risk_stake_flow import emit_cycle_stake_log


def _base_rm(cfg: dict, **overrides):
    rm = SimpleNamespace(
        config={"martingale": {"enabled": True, "multiplier": 2.0}, "orchestrator": {"execution": {}}},
        martingale_config={"enabled": True, "multiplier": 2.0},
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


def test_resolve_martingale_config_defaults_and_clamp():
    assert resolve_martingale_config(None)["enabled"] is False
    assert resolve_martingale_config({})["multiplier"] == pytest.approx(2.0)
    assert resolve_martingale_config({"martingale": {"enabled": True, "multiplier": 0.5}})[
        "multiplier"
    ] == pytest.approx(2.0)
    assert resolve_martingale_config({"martingale": {"enabled": True, "multiplier": "bad"}})[
        "multiplier"
    ] == pytest.approx(2.0)


def test_martingale_enabled_reads_rm_config():
    assert martingale_enabled(SimpleNamespace(martingale_config={"enabled": True})) is True
    assert (
        martingale_enabled(SimpleNamespace(martingale_config=None, config={"martingale": {"enabled": False}})) is False
    )


def test_resolve_martingale_stake_base_and_double():
    rm = SimpleNamespace(
        risk_params={"stake_min": 1.0},
        martingale_config={"enabled": True, "multiplier": 2.0},
        consecutive_losses_linear=0,
        last_loss_stake=0.0,
    )
    stake, tag = resolve_martingale_stake(rm, 100.0)
    assert tag == "MARTINGALE"
    assert stake == pytest.approx(1.0)

    rm.consecutive_losses_linear = 1
    rm.last_loss_stake = 1.0
    assert resolve_martingale_stake(rm, 100.0)[0] == pytest.approx(2.0)

    rm.consecutive_losses_linear = 2
    rm.last_loss_stake = 2.0
    assert resolve_martingale_stake(rm, 100.0)[0] == pytest.approx(4.0)


def test_resolve_martingale_stake_fallback_pow_and_config_lookup():
    rm = SimpleNamespace(
        risk_params={"stake_min": 1.0},
        martingale_config=None,
        config={"martingale": {"enabled": True, "multiplier": 2.0}},
        consecutive_losses_linear=3,
        last_loss_stake=0.0,
    )
    assert resolve_martingale_stake(rm, 100.0)[0] == pytest.approx(8.0)


def test_resolve_martingale_stake_caps_at_bankroll_not_soft_cap():
    rm = SimpleNamespace(
        risk_params={"stake_min": 1.0},
        martingale_config={"enabled": True, "multiplier": 2.0},
        consecutive_losses_linear=3,
        last_loss_stake=8.0,
    )
    assert resolve_martingale_stake(rm, 10.0)[0] == pytest.approx(10.0)


def test_resolve_martingale_stake_zero_when_bankroll_below_min():
    rm = SimpleNamespace(
        risk_params={"stake_min": 1.0},
        martingale_config={"enabled": True, "multiplier": 2.0},
        consecutive_losses_linear=0,
        last_loss_stake=0.0,
    )
    assert resolve_martingale_stake(rm, 0.5)[0] == pytest.approx(0.0)


def test_calculate_stake_martingale_path_bypasses_soft_cap(kelly_config):
    rm = _base_rm(kelly_config, consecutive_losses_linear=2, last_loss_stake=2.0, pending_loss={"R_10": 3.0})
    stake = calculate_stake_for_manager(
        rm,
        bankroll=100.0,
        symbol="R_10",
        conviction=0.7,
        silent=False,
        apply_stop_win=False,
        kwargs={"cycle_id": 1, "dl_metrics": {"execute": True}},
    )
    assert stake == pytest.approx(4.0)
    assert rm._last_stake_audit["mode_tag"] == "MARTINGALE_L2"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"


def test_calculate_stake_hybrid_explore_uses_kelly(kelly_config):
    cfg = dict(kelly_config)
    cfg["kelly"] = {**kelly_config["kelly"], "fraction": 0.25, "max_stake_pct": 0.05}
    rm = _base_rm(cfg, martingale_config={"enabled": True, "multiplier": 2.0})
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


def test_calculate_stake_hybrid_recover_uses_martingale_from_kelly_loss(kelly_config):
    rm = _base_rm(
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
    assert stake == pytest.approx(76.92)
    assert rm._last_stake_audit["mode_tag"] == "MARTINGALE_L1"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"


def test_calculate_stake_martingale_explore_after_reset_uses_kelly(kelly_config):
    cfg = dict(kelly_config)
    cfg["kelly"] = {**kelly_config["kelly"], "fraction": 0.25, "max_stake_pct": 0.05}
    rm = _base_rm(cfg)
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
    rm.logger.info.assert_called()


def test_calculate_martingale_direct_without_dl_metrics(kelly_config):
    rm = _base_rm(kelly_config, consecutive_losses_linear=1, last_loss_stake=1.0)
    stake = calculate_martingale_stake_for_manager(
        rm,
        bankroll=80.0,
        symbol="R_10",
        conviction=0.6,
        silent=False,
        kwargs={"cycle_id": 3},
    )
    assert stake == pytest.approx(2.0)
    assert rm._last_stake_audit["mode_tag"] == "MARTINGALE_L1"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"


def test_emit_cycle_stake_log_martingale_recover_tag():
    rm = SimpleNamespace(_last_stake_audit=None)
    emit_cycle_stake_log(
        rm,
        cycle_id=2,
        silent=False,
        mode_tag="MARTINGALE",
        final_stake=4.0,
        f_star=0.0,
        p=0.55,
        b=0.95,
        bankroll=100.0,
        loss_to_recover=3.0,
        linear_losses=2,
        symbol="R_10",
        rec_info="",
        stake_regime="RECOVER",
        safe_cap=100.0,
        recovery_infeasible=False,
    )
    assert rm._last_stake_audit["mode_tag"] == "MARTINGALE_L2"
    assert rm._last_stake_audit["stake_regime"] == "RECOVER"
