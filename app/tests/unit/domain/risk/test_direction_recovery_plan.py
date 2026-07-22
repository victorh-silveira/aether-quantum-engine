from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.application.services.execution_symbols import symbols_eligible_for_execution
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import apply_soft_recovery_stake
from src.domain.risk.dlambert_sizing import resolve_dlambert_stake
from src.domain.risk.executed_stake_reconciliation import apply_fractional_payoff_residual_to_pending
from src.domain.risk.risk_recovery_state import (
    apply_cluster_profit_to_recovery_state,
    clear_dust_pending_loss,
)
from src.domain.risk.soft_recovery_policy import resolve_soft_recovery_config
from src.domain.risk.stake_sizing import resolve_stake_regime


def test_dust_residual_does_not_reopen_pending_after_win_clears_symbol():
    pending: dict[str, float] = {}
    apply_fractional_payoff_residual_to_pending(pending, "R_10", -0.07)
    assert pending == {}


def test_dust_residual_still_increases_material_pending():
    pending = {"R_10": 5.0}
    apply_fractional_payoff_residual_to_pending(pending, "R_10", -0.07)
    assert pending["R_10"] == pytest.approx(5.07)


def test_clear_dust_pending_returns_false_when_pending_not_dict():
    rm = SimpleNamespace(pending_loss=None, soft_recovery_config={"dust_pending_clear_max": 0.25})
    assert clear_dust_pending_loss(rm) is False


def test_clear_dust_pending_resets_to_explore():
    rm = SimpleNamespace(
        pending_loss={"R_10": 0.07},
        consecutive_losses_linear=3,
        last_loss_stake=2.0,
        soft_recovery_config={"dust_pending_clear_max": 0.25},
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    assert clear_dust_pending_loss(rm) is True
    assert rm.pending_loss == {}
    assert rm.consecutive_losses_linear == 0
    assert rm.last_loss_stake == 0.0
    assert resolve_stake_regime(pending_loss=0.0, consecutive_losses_linear=0) == "EXPLORE"


def test_cluster_win_with_dust_pending_returns_explore():
    rm = SimpleNamespace(
        pending_loss={"R_10": 0.07},
        consecutive_losses_linear=2,
        total_session_profit=8.0,
        last_loss_stake=1.5,
        soft_recovery_config={"dust_pending_clear_max": 0.25},
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    reset = apply_cluster_profit_to_recovery_state(rm, 1.0)
    assert reset is True
    assert rm.pending_loss == {}
    assert rm.consecutive_losses_linear == 0
    assert (
        resolve_stake_regime(
            pending_loss=sum(rm.pending_loss.values()),
            consecutive_losses_linear=rm.consecutive_losses_linear,
        )
        == "EXPLORE"
    )


def test_near_stop_win_freeze_keeps_explore_unit_without_dal_escalation():
    soft = {
        "material_pending_min": 1.0,
        "near_stop_win_freeze_pct": 0.80,
        "max_safe_stake_cap": 4.20,
    }
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=12.0,
        base_unit=1.0,
        consecutive_losses=5,
        previous_stake=3.0,
        bankroll=100.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
        session_pnl=8.5,
        target_win=10.0,
    )
    assert stake == pytest.approx(1.0)
    assert metrics.get("recovery_near_stop_win_freeze") is True
    assert metrics.get("recovery_progression_multiplier") == pytest.approx(1.0)


def test_immaterial_pending_skips_dal_progression():
    soft = {"material_pending_min": 1.0, "near_stop_win_freeze_pct": 0.80}
    metrics: dict = {}
    stake = apply_soft_recovery_stake(
        pending_total=0.40,
        base_unit=1.0,
        consecutive_losses=4,
        previous_stake=2.0,
        bankroll=100.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
        session_pnl=1.0,
        target_win=10.0,
    )
    assert stake == pytest.approx(1.0)
    assert metrics.get("recovery_material_pending") is False


def test_resolve_dlambert_stake_near_target_does_not_escalate_linear():
    class RM:
        dlambert_unit = 1.0
        dlambert_config = {"dlambert_enabled": True}
        soft_recovery_config = {
            "enabled": True,
            "material_pending_min": 1.0,
            "near_stop_win_freeze_pct": 0.80,
            "max_safe_stake_cap": 4.20,
        }
        risk_params = {"payout_estimate": 0.95, "stake_min": 1.0}
        last_loss_stake = 2.0
        total_session_profit = 8.5
        daily_stop_win_target = 10.0

    metrics: dict = {}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=100.0,
        kelly_base=1.0,
        dlambert_config={"dlambert_enabled": True, "soft_recovery": RM.soft_recovery_config},
        rm=RM(),
        consecutive_losses_linear=5,
        pending_total=12.0,
        payout=0.95,
        dl_metrics=metrics,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(1.0)
    assert metrics.get("recovery_near_stop_win_freeze") is True


def test_meta_negative_edge_keeps_weak_call_side():
    metrics = {"calibrated_prob": 0.51, "indicators": {"bb_width": 0.12}}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.12,
        meta_applied=True,
        base_score=0.51,
        symbol="R_10",
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.51)
    assert metrics.get("meta_negative_edge") is True


def test_meta_negative_edge_keeps_direction_without_flip():
    metrics = {"calibrated_prob": 0.51}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.08,
        meta_applied=True,
        base_score=0.51,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.51)
    assert metrics.get("meta_negative_edge") is True


def test_resolve_execution_direction_rejects_meta_negative_edge_without_force():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.51,
            "calibrated_prob": 0.51,
            "val_accuracy": 0.55,
            "predicted_payoff_edge": -0.10,
            "meta_classifier_applied": True,
            "trade_score": 0.51,
        },
    }
    metrics = entry["metrics"]

    def _reject_edge(dl_dir, metrics_arg, predicted_edge, **kwargs):
        metrics_arg["quality_guard_reject"] = True
        metrics_arg["gate_reason"] = "meta_negative_edge"
        return dl_dir, float(kwargs.get("base_score", 0.51))

    with patch(
        "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
        side_effect=_reject_edge,
    ):
        result = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.51,
            -0.10,
            meta_applied=True,
            score=0.51,
            symbol="R_10",
            force=False,
        )
    assert result is None
    assert metrics.get("gate_reason") == "meta_negative_edge"


def test_include_anchor_trades_true_keeps_anchor_eligible():
    symbols = ["R_10", "R_50"]
    eligible = symbols_eligible_for_execution("R_10", symbols, include_anchor=True)
    assert eligible == ["R_10", "R_50"]


def test_soft_recovery_config_exposes_dust_and_freeze_defaults():
    soft = resolve_soft_recovery_config({"soft_recovery": {"enabled": True}})
    assert soft["dust_pending_clear_max"] == pytest.approx(0.25)
    assert soft["near_stop_win_freeze_pct"] == pytest.approx(0.80)
    assert soft["material_pending_min"] == pytest.approx(1.0)
