"""Homologacao do Soft Recovery Adaptativo com teto rigido em micro-banca."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import (
    adaptive_recovery_progression_factor,
    max_safe_stake_cap,
    soft_recovery_progression_multiplier,
)
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.risk_recovery_state import (
    cointegration_pair_score,
    select_cointegration_redirect_candidate,
)
from src.domain.risk.soft_recovery_policy import (
    apply_small_account_hard_floor,
    configured_max_safe_stake_cap,
    configured_max_safe_stake_pct,
    fixed_step_progression_multiplier,
    resolve_amort_cycles,
    resolve_soft_recovery_config,
    soft_recovery_enabled,
)


def _micro_risk_config() -> dict:
    return {
        "kelly": {
            "fraction": 0.005,
            "max_stake_pct": 0.05,
            "max_bankroll_stake_fraction": 0.05,
            "dynamic_win_rate": False,
            "consensus_penalty_enabled": False,
            "stop_win_kelly_enabled": False,
            "recovery_sizing_conviction": 0.50,
            "recovery_min_conviction": 0.50,
            "recovery_min_val_accuracy": 0.53,
        },
        "soft_recovery": {
            "enabled": True,
            "max_safe_stake_cap": 4.20,
            "max_safe_stake_pct": 0.05,
            "amort_cycles_min": 1,
            "amort_cycles_max": 1,
            "cover_multiple": 1.50,
            "coing_redirect_drawdown_threshold": 15.00,
            "infeasible_force_explore": False,
        },
        "params": {"payout_estimate": 0.72, "stake_min": 1.0},
    }


def test_resolve_soft_recovery_config_defaults_match_settings() -> None:
    soft = resolve_soft_recovery_config(None)
    assert soft["enabled"] is True
    assert soft["max_safe_stake_cap"] == pytest.approx(3.0)
    assert soft["max_safe_stake_pct"] == pytest.approx(0.05)
    assert soft["max_safe_stake_pct_linear2"] == pytest.approx(0.04)
    assert soft["max_safe_stake_pct_linear3"] == pytest.approx(0.025)
    assert soft["amort_cycles_min"] == 1
    assert soft["amort_cycles_max"] == 1
    assert soft["cover_multiple"] == pytest.approx(1.5)
    assert soft["linear_bankroll_pct"] == pytest.approx(0.0025)
    assert soft["micro_residual_pending_max"] == pytest.approx(3.0)
    assert soft["micro_residual_zscore_floor"] == pytest.approx(0.01)
    assert soft["micro_residual_gbdt_waiver_skips"] == 4
    assert soft["fixed_step_linear_min"] == 2
    assert soft["fixed_step_linear_max"] == 4
    assert soft["fixed_step_unit_premium"] == pytest.approx(0.12)
    assert soft["small_account_hard_floor_threshold"] == pytest.approx(0.01)
    assert soft["small_account_hard_floor_pct"] == pytest.approx(0.01)
    assert soft["dust_pending_clear_max"] == pytest.approx(0.25)
    assert soft["near_stop_win_freeze_pct"] == pytest.approx(0.70)
    assert soft["material_pending_min"] == pytest.approx(0.5)
    assert soft["infeasible_force_explore"] is True
    assert soft["pending_waives_scale_explore"] is True
    assert soft["adapted_force_explore"] is True
    assert soft["adapted_force_explore_linear_min"] == 2
    assert soft["live_evidence_force_explore_linear_min"] == 3
    assert soft["live_evidence_force_explore_n_min"] == 2
    assert soft["live_evidence_force_explore_wr_max"] == pytest.approx(0.62)


def test_resolve_amort_cycles_bounds_from_soft_recovery() -> None:
    soft = {"amort_cycles_min": 2, "amort_cycles_max": 5}
    assert resolve_amort_cycles(1, soft) == 4
    assert resolve_amort_cycles(2, soft) == 3
    assert resolve_amort_cycles(5, soft) == 2
    full = {"amort_cycles_min": 1, "amort_cycles_max": 1}
    assert resolve_amort_cycles(1, full) == 1
    assert resolve_amort_cycles(8, full) == 1


def test_soft_recovery_clipping_linear_five_micro_bank_is_exactly_four_twenty() -> None:
    rm = RiskManager(_micro_risk_config())
    rm.initial_bankroll = 100.0
    rm.dlambert_unit = 1.0
    rm.consecutive_losses_linear = 5
    rm.pending_loss = {"R_10": 22.0}
    rm.last_loss_stake = 8.0
    rm.total_session_profit = -22.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.62)
    rm._recovery_allowed = MagicMock(return_value=True)

    payout = 0.95
    unit = 1.0
    geometric_adaptive = unit * soft_recovery_progression_multiplier(5, payout=payout)
    classical_two_pow = unit * (2**5)
    assert geometric_adaptive > 4.20
    assert classical_two_pow == pytest.approx(32.0)

    stake = rm.calculate_stake(
        100.0,
        "R_10",
        0.70,
        silent=True,
        apply_stop_win=False,
        dl_metrics={
            "execute": True,
            "trade_score": 0.70,
            "val_accuracy": 0.66,
            "raw_prob": 0.70,
            "calibrated_prob": 0.70,
        },
    )
    assert stake == pytest.approx(4.20)
    assert stake < classical_two_pow
    assert stake < geometric_adaptive
    assert max_safe_stake_cap(
        100.0,
        consecutive_losses_linear=5,
        soft_recovery=rm.soft_recovery_config,
    ) == pytest.approx(4.20)
    assert adaptive_recovery_progression_factor(0.95) == pytest.approx(1.0 + 1.0 / 0.95)


def test_soft_recovery_disabled_falls_back_to_kelly() -> None:
    cfg = _micro_risk_config()
    cfg["soft_recovery"]["enabled"] = False
    rm = RiskManager(cfg)
    rm.initial_bankroll = 100.0
    rm.dlambert_unit = 1.0
    rm.consecutive_losses_linear = 5
    rm.pending_loss = {"R_10": 22.0}
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.62)
    rm._recovery_allowed = MagicMock(return_value=True)
    stake = rm.calculate_stake(
        100.0,
        "R_10",
        0.70,
        silent=True,
        apply_stop_win=False,
        dl_metrics={"execute": True, "trade_score": 0.70, "val_accuracy": 0.66},
    )
    assert stake <= 3.50
    assert stake != pytest.approx(4.20)


def test_fixed_step_and_hard_floor_helpers() -> None:
    assert fixed_step_progression_multiplier(2) == pytest.approx(1.12)
    assert fixed_step_progression_multiplier(3) == pytest.approx(1.12)
    assert fixed_step_progression_multiplier(4) == pytest.approx(1.12)
    assert fixed_step_progression_multiplier(5) is None
    assert apply_small_account_hard_floor(4.20, 80.0) == pytest.approx(4.20)
    assert apply_small_account_hard_floor(4.20, 100.0) == pytest.approx(4.20)
    assert apply_small_account_hard_floor(1.0, 0.0) == pytest.approx(1.0)


def test_soft_recovery_policy_branches_and_tail_cap() -> None:
    assert soft_recovery_enabled(soft_recovery={"max_safe_stake_cap": 4.20}) is True
    assert soft_recovery_enabled({"soft_recovery": {"enabled": False}}) is False
    assert soft_recovery_enabled({"dlambert_enabled": False}) is False
    assert configured_max_safe_stake_cap(None) is None
    assert configured_max_safe_stake_cap({}) is None
    assert configured_max_safe_stake_cap({"max_safe_stake_cap": "bad"}) is None
    assert configured_max_safe_stake_cap({"max_safe_stake_cap": -1.0}) is None
    assert configured_max_safe_stake_cap({"max_safe_stake_cap": 4.20}) == pytest.approx(4.20)
    assert configured_max_safe_stake_pct(None) == pytest.approx(0.05)
    assert configured_max_safe_stake_pct({"max_safe_stake_pct": 0.035}) == pytest.approx(0.035)
    assert configured_max_safe_stake_pct({"max_safe_stake_pct": "bad"}) == pytest.approx(0.05)
    assert configured_max_safe_stake_pct({"max_safe_stake_pct": -1.0}) == pytest.approx(0.05)
    assert configured_max_safe_stake_pct({"max_safe_stake_pct": 2.0}) == pytest.approx(1.0)
    rm = RiskManager(_micro_risk_config())
    rm.initial_bankroll = 100.0
    assert rm.max_safe_tail_cap() == pytest.approx(4.20)
    assert rm.max_safe_tail_cap(1000.0) == pytest.approx(84.0)
    rm.initial_bankroll = 0.0
    assert rm.max_safe_tail_cap() == pytest.approx(4.20)
    assert soft_recovery_enabled({"soft_recovery": {"amort_cycles_min": 2}}) is True
    assert resolve_soft_recovery_config(None)["enabled"] is True
    assert resolve_soft_recovery_config({"soft_recovery": "bad"})["max_safe_stake_cap"] == pytest.approx(3.0)
    assert cointegration_pair_score({"calibrated_prob": 0.7, "edge_zscore": -0.1}) == float("-inf")
    assert cointegration_pair_score({"calibrated_prob": 0.8, "edge_zscore": 1.5}) > 0.0
    assert select_cointegration_redirect_candidate([("R_50", TradeDirection.CALL, {"edge_zscore": 2.0})]) == []
    alone = [("stp_500", TradeDirection.CALL, {"calibrated_prob": 0.7, "edge_zscore": 1.0})]
    assert select_cointegration_redirect_candidate(alone) == alone
    pair = [
        ("stp_500", TradeDirection.CALL, {"calibrated_prob": 0.55, "edge_zscore": 0.4}),
        ("R_50", TradeDirection.PUT, {"calibrated_prob": 0.80, "edge_zscore": 1.5}),
    ]
    assert select_cointegration_redirect_candidate(pair)[0][0] == "stp_500"
    multi = [
        ("stp_500", TradeDirection.CALL, {"calibrated_prob": 0.55, "edge_zscore": 0.4}),
        ("R_50", TradeDirection.PUT, {"calibrated_prob": 0.80, "edge_zscore": 1.5}),
    ]
    with patch(
        "src.domain.risk.risk_recovery_state.DRIFT_PAIR_SYMBOLS",
        frozenset({"stp_500", "R_50"}),
    ):
        assert select_cointegration_redirect_candidate(multi)[0][0] == "R_50"
