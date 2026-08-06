"""Testes de Kelly bayesiano e recovery (parte 2)."""

from unittest.mock import MagicMock

import pytest

from src.domain.risk.bayesian_win_rate import bayesian_win_rate
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.soft_recovery_policy import is_recovery_infeasible
from src.domain.risk.stake_sizing import compute_single_strike_kelly_base


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


def test_recover_weak_score_does_not_force_mandatory_min(kelly_config):
    rm = _rm(kelly_config)
    rm.pending_loss = {}
    rm.consecutive_losses_linear = 1
    rm.soft_recovery_config = {"enabled": False}
    rm.dlambert_config = {"dlambert_enabled": False, "soft_recovery": {"enabled": False}}
    rm._recovery_allowed = MagicMock(return_value=True)
    metrics = {"execute": False, "trade_score": 0.40, "raw_prob": 0.40}
    stake_mandatory = calculate_stake_for_manager(
        rm,
        120.0,
        "OTC_SPC",
        0.40,
        silent=True,
        apply_stop_win=False,
        kwargs={
            "dl_metrics": dict(metrics),
            "mandatory_weak_cap": True,
            "mandatory_trade_each_cycle": True,
        },
    )
    stake_plain = calculate_stake_for_manager(
        rm,
        120.0,
        "OTC_SPC",
        0.40,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": dict(metrics)},
    )
    assert stake_mandatory == pytest.approx(stake_plain)


def test_explore_emits_stake_regime_on_metrics(kelly_config):
    rm = _rm(kelly_config)
    metrics = {"execute": True, "trade_score": 0.62, "raw_prob": 0.62}
    calculate_stake_for_manager(
        rm,
        120.0,
        "OTC_SPC",
        0.62,
        silent=False,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "cycle_id": 3},
    )
    assert metrics.get("stake_regime") == "EXPLORE"
    assert rm._last_stake_audit["mode_tag"].startswith("EXPLORE_")


def test_recovery_infeasible_edge_cases():
    soft = {"amort_cycles_max": 5}
    assert is_recovery_infeasible(0.0, 4.20, 0.95, soft) is False
    assert is_recovery_infeasible(10.0, 0.0, 0.95, soft) is True
    assert is_recovery_infeasible(10.0, 4.20, 0.0, soft) is True


def test_bayesian_win_rate_handles_bad_live_fields():
    p = bayesian_win_rate(
        0.60,
        metrics={
            "live_n": 32,
            "live_wr": "bad",
            "live_brier": "x",
            "live_ece": object(),
            "edge_zscore": "z",
        },
    )
    assert 0.40 <= p <= 0.75
    shrunk = bayesian_win_rate(
        0.65,
        metrics={"live_n": 40, "live_wr": 0.70, "live_brier": 0.30, "live_ece": 0.15, "edge_zscore": 2.0},
    )
    assert shrunk < 0.70


def test_stop_win_kelly_gate_blocks_without_live_health():
    base = compute_single_strike_kelly_base(
        10.0,
        1000.0,
        0.95,
        0.80,
        {"large_account_stop_win_pct": 10.0, "small_account_threshold": 50.0},
        {"stop_win_kelly_enabled": True, "stop_win_kelly_max_fraction": 1.0},
        1000.0,
        0.0,
        has_active_contracts=False,
        live_metrics={"live_n": 10, "live_wr": 0.40},
    )
    assert base == pytest.approx(10.0)
    bad_wr = compute_single_strike_kelly_base(
        10.0,
        1000.0,
        0.95,
        0.80,
        {"large_account_stop_win_pct": 10.0, "small_account_threshold": 50.0},
        {"stop_win_kelly_enabled": True, "stop_win_kelly_max_fraction": 1.0},
        1000.0,
        0.0,
        has_active_contracts=False,
        live_metrics={"live_n": 50, "live_wr": object()},
    )
    assert bad_wr == pytest.approx(10.0)
    done = compute_single_strike_kelly_base(
        10.0,
        1000.0,
        0.95,
        0.80,
        {"large_account_stop_win_pct": 10.0, "small_account_threshold": 50.0},
        {"stop_win_kelly_enabled": True, "stop_win_kelly_max_fraction": 1.0},
        1000.0,
        1000.0,
        has_active_contracts=False,
        live_metrics={"live_n": 50, "live_wr": 0.60},
    )
    assert done == pytest.approx(10.0)
