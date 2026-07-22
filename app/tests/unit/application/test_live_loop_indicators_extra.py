from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import market_decision_score
from src.application.services.live_signal_metrics import (
    _ece,
    apply_live_calib_drift_soft,
    live_signal_snapshot,
    record_live_signal_outcome,
    reset_live_signal_metrics,
)
from src.application.services.meta_payoff_shadow import (
    record_meta_payoff_shadow_pair,
    reset_meta_payoff_shadow,
)
from src.application.services.meta_payoff_veto_gate import should_veto_meta_payoff_negative_zscore
from src.application.services.orchestrator.settlement_logic import _process_contract_outcome
from src.domain.models.trade import TradeDirection


def test_live_ece_empty_and_unknown_direction():
    assert _ece([], []) == 0.0
    orch = SimpleNamespace()
    reset_live_signal_metrics(orch)
    record_live_signal_outcome(orch, "X", won=True, raw_prob=None, direction="FLAT")
    snap = live_signal_snapshot(orch, "X")
    assert snap["live_n"] == 1
    assert snap["live_wr"] == 1.0


def test_calib_drift_falls_back_to_raw_side_score():
    metrics = {
        "live_n": 48,
        "live_ece": 0.20,
        "live_wr": 0.40,
        "raw_prob": 0.85,
    }
    assert apply_live_calib_drift_soft(metrics) is True
    assert metrics["trade_score"] == pytest.approx(max(0.48, 0.85 * 0.72))


def test_calib_drift_falls_back_to_resolved_conviction():
    metrics = {
        "live_n": 48,
        "live_ece": 0.20,
        "live_wr": 0.40,
        "raw_prob": 0.85,
        "resolved_conviction": 0.77,
    }
    assert apply_live_calib_drift_soft(metrics) is True
    assert metrics["trade_score"] == pytest.approx(max(0.48, 0.77 * 0.72))


def test_market_rank_mid_ece_penalty():
    base = {
        "raw_prob": 0.80,
        "val_accuracy": 0.60,
        "edge": 0.30,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.12,
        "live_n": 0,
        "val_ece": 0.09,
    }
    clean = dict(base)
    clean.pop("val_ece")
    assert market_decision_score(base) < market_decision_score(clean)


def test_hard_veto_path_returns_none_from_resolver():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    for i in range(64):
        record_meta_payoff_shadow_pair(z_score=float(i) * 0.1, profit=float(i), orch=orch)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.80,
            "val_accuracy": 0.70,
            "predicted_payoff_edge": 0.08,
            "edge_expectancy": "LOSS_EXPECTED",
            "meta_payoff_edge_zscore": -0.90,
            "edge_zscore": -0.90,
            "edge_zscore_samples": 20,
            "edge_zscore_window": 15,
            "trade_score": 0.80,
        },
    }
    result = resolve_execution_direction(entry, symbol="R_10", orch=orch)
    assert result is None
    assert entry["metrics"].get("gate_reason") == "meta_payoff_negative_zscore_veto"
    reset_meta_payoff_shadow()


def test_waiver_exception_falls_back_to_hard_when_shadow_ready():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    for i in range(64):
        record_meta_payoff_shadow_pair(z_score=float(i) * 0.1, profit=float(i), orch=orch)
    rm = SimpleNamespace(
        consecutive_losses_linear=0,
        pending_loss={},
        pending_loss_total=lambda: 0.0,
        initial_bankroll=100.0,
    )
    metrics = {
        "predicted_payoff_edge": -0.05,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": -0.90,
        "edge_zscore": -0.90,
        "edge_zscore_samples": 20,
        "edge_zscore_window": 15,
        "raw_prob": 0.80,
    }
    with patch(
        "src.application.services.meta_payoff_veto_gate.meta_payoff_veto_emergency_waiver",
        side_effect=RuntimeError("boom"),
    ):
        hard = should_veto_meta_payoff_negative_zscore(
            metrics,
            direction=TradeDirection.CALL,
            risk_manager=rm,
            orch=orch,
        )
    assert hard is True
    assert metrics.get("meta_veto_mode") == "hard"
    reset_meta_payoff_shadow()


def test_process_contract_outcome_uses_audit_direction():
    rm = SimpleNamespace(
        contract_to_symbol={},
        contract_requested_stakes={},
        contract_stakes={},
        active_contract_ids=[],
        register_result=lambda *a, **k: None,
    )
    orch = SimpleNamespace(
        state=SimpleNamespace(balance=100.0),
        risk_manager=rm,
        tick_count=1,
        _cluster_results=[],
        _contract_cycle={},
        _session_wins=0,
        _session_losses=0,
        config={"deep_learning": {}},
    )
    with (
        patch(
            "src.application.services.orchestrator.settlement_outcome.resolve_executed_buy_stake",
            return_value=1.0,
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.reconcile_settlement_profit",
            side_effect=lambda p, *_a, **_k: p,
        ),
        patch("src.application.services.orchestrator.settlement_outcome.bind_executed_stake_for_contract"),
        patch("src.application.services.orchestrator.settlement_outcome.record_symbol_outcome"),
        patch("src.application.services.orchestrator.settlement_outcome.record_direction_outcome"),
        patch("src.application.services.orchestrator.settlement_outcome.record_live_signal_outcome") as live,
        patch("src.application.services.orchestrator.settlement_outcome.record_side_equilibrium_outcome") as side_eq,
        patch("src.application.services.orchestrator.settlement_outcome.mark_force_retrain"),
        patch("src.application.services.orchestrator.settlement_logic.log_cluster_summary"),
    ):
        _process_contract_outcome(
            orch,
            {"underlying": "R_10"},
            None,
            7,
            -1.0,
            audit_direction="PUT",
            audit_raw_prob=0.22,
        )
    live.assert_called_once()
    assert live.call_args.kwargs["direction"] == "PUT"
    side_eq.assert_called_once()
    assert side_eq.call_args.kwargs["direction"] == "PUT"
    assert side_eq.call_args.kwargs["won"] is False
    assert orch._last_loss_direction == "PUT"
