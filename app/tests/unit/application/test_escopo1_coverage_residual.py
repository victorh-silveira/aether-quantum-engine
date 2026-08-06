"""Cobertura residual ampla apos remocao dos vetos."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.execution_quality_gate_margin import stamp_edge_without_direction
from src.application.services.force_trade_mode import synthesize_force_direction, synthesize_force_trade_candidate
from src.application.services.live_signal_metrics import (
    apply_live_calib_drift_soft,
    attach_live_signal_metrics,
    live_signal_snapshot,
    record_live_signal_outcome,
    reset_live_signal_metrics,
)
from src.application.services.live_signal_metrics_config import load_live_signal_metrics_from_settings
from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome, snapshot_side_counts
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import (
    cointegration_valve_suppressed,
    gbdt_waiver_skip_threshold_for_risk,
    is_low_intensity_recovery,
    is_micro_residual_liability,
    negative_zscore_veto_floor_for_risk,
    resolve_gbdt_waiver_skip_threshold,
    resolve_negative_zscore_veto_floor,
    risk_session_bankroll_pending,
)


def test_live_signal_metrics_roundtrip():
    orch = SimpleNamespace(_live_signal_metrics=None, _active_cycle_id=3)
    reset_live_signal_metrics(orch)
    assert live_signal_snapshot(orch, "OTC_SPC")["live_n"] == 0
    for i in range(24):
        record_live_signal_outcome(orch, "OTC_SPC", won=i % 2 == 0, raw_prob=0.62, direction="CALL")
    snap = live_signal_snapshot(orch, "OTC_SPC")
    assert snap["live_n"] == 24
    metrics = {"raw_prob": 0.9, "live_n": 40, "live_ece": 0.9, "live_wr": 0.1, "trade_score": 0.8}
    attach_live_signal_metrics(orch, "OTC_SPC", metrics)
    from unittest.mock import patch

    with (
        patch(
            "src.application.services.live_signal_metrics.load_sample_size_policy",
            return_value={"calib_soft_min_n": 1},
        ),
        patch(
            "src.application.services.live_signal_metrics._live",
            return_value={
                "ece_soft_threshold": 0.05,
                "drift_soft_penalty": 0.1,
                "drift_soft_veto_n": 10,
                "drift_min_score": 0.4,
                "drift_score_factor": 0.5,
                "window": 64,
                "min_rank": 8,
                "ece_bins": 10,
            },
        ),
    ):
        assert apply_live_calib_drift_soft(metrics, orch=orch, symbol="OTC_SPC") is True
    assert load_live_signal_metrics_from_settings()["window"] >= 1


def test_soft_recovery_policy_branches():
    soft = {
        "enabled": True,
        "micro_residual_bankroll_max": 200.0,
        "micro_residual_pending_max": 5.0,
        "micro_residual_pending_pct": 0.1,
        "micro_residual_zscore_floor": -0.5,
        "negative_zscore_veto": -0.2,
        "micro_residual_gbdt_waiver_skips": 2,
        "gbdt_waiver_skip_cycles": 4,
    }
    assert is_micro_residual_liability(100.0, 1.0, soft_recovery=soft) is True
    assert is_micro_residual_liability(0.0, 1.0, soft_recovery=soft) is False
    assert is_micro_residual_liability(100.0, 0.0, soft_recovery=soft) is False
    assert is_micro_residual_liability(500.0, 1.0, soft_recovery=soft) is False
    assert is_micro_residual_liability(100.0, 9.0, soft_recovery=soft) is False
    assert is_low_intensity_recovery(100.0, 1.0, soft_recovery=soft) is True
    assert resolve_negative_zscore_veto_floor(100.0, 1.0, soft_recovery=soft) == pytest.approx(-0.5)
    assert resolve_negative_zscore_veto_floor(1000.0, 1.0, soft_recovery=soft) == pytest.approx(-0.2)
    assert resolve_gbdt_waiver_skip_threshold(100.0, 1.0, soft_recovery=soft) == 2
    assert resolve_gbdt_waiver_skip_threshold(1000.0, 1.0, soft_recovery=soft) == 4
    assert cointegration_valve_suppressed(100.0, 1.0, soft_recovery=soft) is True
    rm = SimpleNamespace(initial_bankroll=100.0, pending_loss_total=lambda: 1.0, soft_recovery_config=soft)
    assert risk_session_bankroll_pending(rm)[0] == pytest.approx(100.0)
    assert negative_zscore_veto_floor_for_risk(rm) == pytest.approx(-0.5)
    assert gbdt_waiver_skip_threshold_for_risk(rm) == 2
    assert negative_zscore_veto_floor_for_risk(None) < 0.0
    assert gbdt_waiver_skip_threshold_for_risk(None) >= 1
    rm2 = SimpleNamespace(initial_bankroll=0.0, pending_loss={}, soft_recovery_config=None)
    assert negative_zscore_veto_floor_for_risk(rm2) < 0.0
    assert gbdt_waiver_skip_threshold_for_risk(rm2) >= 1


def test_side_eq_store_persist_paths():
    client = MagicMock()
    client.hset = MagicMock(return_value=1)
    writer = MagicMock()
    writer.enqueue_trade_outcome = MagicMock(return_value=None)
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        state_store=SimpleNamespace(client=client),
        timescale_writer=writer,
    )
    record_side_equilibrium_outcome(orch, "OTC_SPC", direction="PUT", won=False, profit=-1.0, raw_prob=0.4)
    assert snapshot_side_counts(orch, "OTC_SPC", window=10).put_n >= 1
    client.hset.side_effect = RuntimeError("boom")
    record_side_equilibrium_outcome(orch, "OTC_SPC", direction="CALL", won=True)
    writer.enqueue_trade_outcome.side_effect = RuntimeError("ts")
    record_side_equilibrium_outcome(orch, "OTC_SPC", direction="CALL", won=True)
    assert record_side_equilibrium_outcome(orch, "OTC_SPC", direction="HOLD", won=True).call_n >= 0


def test_market_rank_penalties_and_recovery():
    metrics = {
        "raw_prob": 0.7,
        "val_accuracy": 0.6,
        "edge": 0.2,
        "live_n": 40,
        "live_brier": 0.4,
        "live_ece": 0.3,
        "live_wr": 0.4,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.01,
        "meta_squeeze_downgrade": True,
        "indicators": {"adx": 0.1, "vol_ratio": 0.5, "hurst": 0.7},
    }
    score = market_decision_score(metrics, recovery_active=True, symbol="OTC_SPC", last_loss_symbol="OTC_SPC")
    assert isinstance(score, float)
    metrics2 = dict(metrics)
    metrics2["indicators"] = {"adx": 0.5, "vol_ratio": 1.2, "hurst": 0.3}
    assert isinstance(
        market_decision_score(metrics2, recovery_active=True, symbol="R_50", last_loss_symbol="OTC_SPC"), float
    )


def test_force_trade_more_branches():
    assert (
        synthesize_force_direction({"direction": TradeDirection.PUT, "metrics": {"deploy_ok": True}})
        == TradeDirection.PUT
    )
    assert synthesize_force_direction({"metrics": {"calibrated_prob": "bad", "deploy_ok": True}}) is None
    assert synthesize_force_trade_candidate(["OTC_SPC"], {"OTC_SPC": "bad"}) is None
    assert synthesize_force_trade_candidate(["OTC_SPC"], {"OTC_SPC": {"metrics": {"deploy_ok": False}}}) is None
    assert synthesize_force_trade_candidate([], {}) is None


def test_stamp_edge_without_base_score():
    metrics = {"predicted_payoff_edge": 0.2}
    stamp_edge_without_direction(metrics, margin_floor=0.1)
    assert metrics["edge_without_direction"] is True
    linear, pending = read_risk_session_state(SimpleNamespace(pending_loss_total=lambda: 3.0), linear=4)
    assert linear == 4
    assert pending == pytest.approx(3.0)
