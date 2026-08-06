"""Casos extras de recovery infeasible e emit de stake regime."""

from unittest.mock import MagicMock

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.risk_stake_flow import emit_cycle_stake_log


def _rm(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "mandatory_weak_conviction_cap": 0.55,
        "mandatory_weak_max_stake_pct": 0.01,
        "stop_win_kelly_enabled": False,
        "fraction": 0.005,
        "max_stake_pct": 0.035,
    }
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0}
    rm.soft_recovery_config = kelly_config.get("soft_recovery", {})
    rm.dlambert_config = kelly_config.get("dlambert", {})
    rm.initial_bankroll = 120.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 1.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._recovery_allowed = MagicMock(return_value=False)
    return rm


def test_emit_invalid_regime_defaults_explore(kelly_config):
    rm = MagicMock()
    emit_cycle_stake_log(
        rm,
        cycle_id=9,
        silent=False,
        mode_tag="KELLY",
        final_stake=1.0,
        f_star=0.01,
        p=0.55,
        b=0.95,
        bankroll=100.0,
        loss_to_recover=0.0,
        linear_losses=0,
        symbol="OTC_SPC",
        rec_info="",
        stake_regime="WEIRD",
        safe_cap=3.5,
        recovery_infeasible=False,
    )
    assert rm._last_stake_audit["mode_tag"] == "EXPLORE_KELLY"


def test_recover_mandatory_blocked_returns_zero_when_below_min(kelly_config):
    rm = _rm(kelly_config)
    rm.pending_loss = {}
    rm.consecutive_losses_linear = 1
    rm.soft_recovery_config = {"enabled": False}
    rm.dlambert_config = {"dlambert_enabled": False, "soft_recovery": {"enabled": False}}
    rm._recovery_allowed = MagicMock(return_value=False)
    metrics = {"execute": False, "trade_score": 0.40, "raw_prob": 0.50}
    stake = calculate_stake_for_manager(
        rm,
        120.0,
        "OTC_SPC",
        0.40,
        silent=True,
        apply_stop_win=False,
        kwargs={
            "dl_metrics": metrics,
            "mandatory_weak_cap": True,
            "mandatory_trade_each_cycle": True,
        },
    )
    assert stake == 0.0


def test_recovery_infeasible_logs_when_not_silent(kelly_config):
    rm = _rm(kelly_config)
    rm.pending_loss = {"OTC_SPC": 80.0}
    rm.consecutive_losses_linear = 2
    rm.dlambert_unit = 1.0
    rm._recovery_allowed = MagicMock(return_value=True)
    rm.soft_recovery_config = {
        "enabled": True,
        "max_safe_stake_cap": 4.20,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "infeasible_force_explore": True,
    }
    metrics = {"execute": True, "trade_score": 0.70, "raw_prob": 0.70}
    calculate_stake_for_manager(
        rm,
        90.0,
        "OTC_SPC",
        0.70,
        silent=False,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "cycle_id": 11},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "RECOVERY_INFEASIBLE" in logged or metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("stake_regime") == "EXPLORE"


def test_cross_veto_waiver_and_mandatory_weak_zero_pct_residual():
    from src.domain.risk.consensus_stake_penalty import cross_veto_recovery_waiver_allowed
    from src.domain.risk.risk_stake_calc import _apply_mandatory_weak_explore_cap
    from src.domain.risk.soft_recovery_config import (
        load_soft_recovery_from_settings,
        reset_soft_recovery_config_cache,
        resolve_soft_recovery_config,
    )

    rm = type("RM", (), {"consecutive_losses_linear": 5, "pending_loss": {"OTC_SPC": 260.0}})()
    assert cross_veto_recovery_waiver_allowed({"raw_prob": 0.82}, direction="CALL", risk_manager=rm) is True
    assert (
        _apply_mandatory_weak_explore_cap(
            10.0,
            100.0,
            stake_regime="EXPLORE",
            mandatory_flag=True,
            dl_execute=False,
            kelly_config={"mandatory_weak_max_stake_pct": 0.0},
        )
        == 10.0
    )
    reset_soft_recovery_config_cache()
    base = load_soft_recovery_from_settings()
    flat = resolve_soft_recovery_config({"enabled": True, "max_safe_stake_cap": base["max_safe_stake_cap"]})
    assert flat["infeasible_force_explore"] is True
