from types import SimpleNamespace

import pytest

from src.application.services.execution_quality_gate import (
    MANDATORY_MIN_TRADE_SCORE_DEFAULT,
    MIN_DIRECTION_MARGIN_DEFAULT,
    MIN_PAYOFF_EDGE_DEFAULT,
    REGULAR_MIN_DIRECTION_MARGIN_DEFAULT,
    REGULAR_MIN_PAYOFF_EDGE_DEFAULT,
    apply_quality_penalty_to_metrics,
    direction_margin_from_probability,
    ensure_direction_margin,
    format_quality_guard_log_message,
    passes_execution_quality,
    quality_gate_params,
    read_risk_session_state,
    resolve_dynamic_quality_limits,
    sync_direction_margin,
)
from src.application.services.execution_quality_gate_cluster import (
    log_quality_guard_suspension,
    quality_conviction_suspends_cluster,
)


def _edge_signal_metrics() -> dict:
    return {
        "calibrated_prob": 0.57,
        "predicted_payoff_edge": 0.02,
        "meta_classifier_applied": True,
    }


def test_quality_gate_params_custom_config():
    params = quality_gate_params(
        {
            "quality_gate": {
                "min_direction_margin": 0.08,
                "min_payoff_edge": 0.05,
                "inverted_min_score": 0.80,
                "min_adx_normal": 0.20,
                "regular": {"min_direction_margin": 0.05, "min_payoff_edge": 0.02},
            }
        }
    )
    assert params["min_direction_margin"] == 0.08
    assert params["min_payoff_edge"] == 0.05
    assert params["inverted_min_score"] == 0.80
    assert params["min_adx_normal"] == 0.20


def test_quality_gate_params_defaults():
    params = quality_gate_params({})
    assert params["min_direction_margin"] == MIN_DIRECTION_MARGIN_DEFAULT
    assert params["min_payoff_edge"] == MIN_PAYOFF_EDGE_DEFAULT
    assert pytest.approx(0.72) == MANDATORY_MIN_TRADE_SCORE_DEFAULT


def test_resolve_dynamic_quality_limits_regular_regime():
    limits = resolve_dynamic_quality_limits({}, linear=0, pending_loss_total=0.0)
    assert limits["quality_regime"] == "regular"
    assert limits["min_direction_margin"] == REGULAR_MIN_DIRECTION_MARGIN_DEFAULT
    assert limits["min_payoff_edge"] == REGULAR_MIN_PAYOFF_EDGE_DEFAULT


def test_resolve_dynamic_quality_limits_recovery_regime():
    limits = resolve_dynamic_quality_limits({}, linear=2, pending_loss_total=0.0)
    assert limits["quality_regime"] == "recovery"
    assert limits["min_direction_margin"] == MIN_DIRECTION_MARGIN_DEFAULT
    assert limits["min_payoff_edge"] == MIN_PAYOFF_EDGE_DEFAULT


def test_read_risk_session_state_from_manager():
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=3,
        pending_loss={"RDBULL": 4.5, "RDBEAR": 1.5},
        pending_loss_total=lambda: 6.0,
    )
    linear, pending = read_risk_session_state(risk_manager)
    assert linear == 3
    assert pending == 6.0


def test_passes_execution_quality_regular_regime_accepts_elastic_signal():
    metrics = _edge_signal_metrics()
    assert passes_execution_quality(metrics, linear=0, pending_loss_total=0.0) is True
    assert metrics["quality_gate_regime"] == "regular"
    assert metrics["regime_skip_cycle"] is False
    assert "quality_gate_reason" not in metrics


def test_passes_execution_quality_recovery_regime_rejects_same_signal():
    metrics = _edge_signal_metrics()
    assert passes_execution_quality(metrics, linear=2, pending_loss_total=0.0) is False
    assert metrics["quality_gate_regime"] == "recovery"
    assert metrics["regime_skip_cycle"] is True
    reason = metrics["quality_gate_reason"]
    assert "Payoff" in reason
    assert "<" in reason
    assert "min" in reason


def test_quality_gate_params_ignores_non_dict_config():
    assert quality_gate_params({"quality_gate": "invalid"})["min_direction_margin"] == MIN_DIRECTION_MARGIN_DEFAULT
    assert quality_gate_params("nope")["min_payoff_edge"] == MIN_PAYOFF_EDGE_DEFAULT


def test_read_risk_session_state_sums_pending_loss_dict():
    risk_manager = SimpleNamespace(consecutive_losses_linear=0, pending_loss={"RDBULL": 2.5, "RDBEAR": 1.5})
    linear, pending = read_risk_session_state(risk_manager)
    assert linear == 0
    assert pending == 4.0


def test_ensure_direction_margin_from_probability():
    metrics = {"calibrated_prob": 0.70}
    assert ensure_direction_margin(metrics) == pytest.approx(0.20)


def test_ensure_direction_margin_uses_put_side_probability():
    metrics = {"calibrated_prob": 0.46, "dl_direction": "PUT"}
    assert ensure_direction_margin(metrics) == pytest.approx(0.04)


def test_sync_direction_margin_falls_back_to_lateral_scores():
    metrics = {
        "direction_call_score": 0.72,
        "direction_put_score": 0.28,
        "exec_direction": "CALL",
    }
    assert sync_direction_margin(metrics, direction="CALL") == pytest.approx(0.44)


def test_direction_margin_from_probability_call_high_conviction():
    assert direction_margin_from_probability(0.75, direction="CALL") == pytest.approx(0.25)


def test_passes_execution_quality_rejects_low_margin_in_recovery():
    metrics = {"calibrated_prob": 0.55, "predicted_payoff_edge": 0.10, "meta_classifier_applied": True}
    assert passes_execution_quality(metrics, linear=1, pending_loss_total=0.0) is False
    assert metrics["regime_skip_cycle"] is True
    reason = metrics["quality_gate_reason"]
    assert "TCN Margin" in reason
    assert "<" in reason
    assert "min" in reason
    assert "Payoff" not in reason


def test_passes_execution_quality_rejects_low_edge_in_recovery():
    metrics = {
        "calibrated_prob": 0.70,
        "predicted_payoff_edge": 0.01,
        "meta_classifier_applied": True,
    }
    assert passes_execution_quality(metrics, linear=0, pending_loss_total=1.0) is False
    assert metrics["regime_skip_cycle"] is True
    reason = metrics["quality_gate_reason"]
    assert "Meta Payoff" in reason
    assert "<" in reason
    assert "min" in reason


def test_passes_execution_quality_ignores_edge_without_meta_classifier():
    metrics = {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.0}
    assert passes_execution_quality(metrics, linear=2, pending_loss_total=0.0) is True
    assert "quality_gate_reason" not in metrics


def test_passes_execution_quality_margin_reject_without_meta_has_no_payoff_clause():
    metrics = {"calibrated_prob": 0.55}
    assert passes_execution_quality(metrics, linear=2, pending_loss_total=0.0) is False
    reason = metrics["quality_gate_reason"]
    assert "TCN Margin" in reason
    assert "Payoff" not in reason
    assert "None" not in reason
    assert "null" not in reason.lower()


def test_passes_execution_quality_accepts_high_conviction_in_recovery():
    metrics = {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.06, "meta_classifier_applied": True}
    assert passes_execution_quality(metrics, linear=2, pending_loss_total=0.0) is True
    assert metrics["regime_skip_cycle"] is False


def test_apply_quality_penalty_returns_unit_penalty_on_reject():
    metrics = {"calibrated_prob": 0.52, "predicted_payoff_edge": 0.06, "meta_classifier_applied": True}
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    penalty = apply_quality_penalty_to_metrics(metrics, risk_manager=risk_manager)
    assert penalty == 1.0


def test_apply_quality_penalty_returns_zero_on_pass():
    metrics = {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.06, "meta_classifier_applied": True}
    assert apply_quality_penalty_to_metrics(metrics, linear=0, pending_loss_total=0.0) == 0.0


def test_format_quality_guard_log_message_structure():
    message = format_quality_guard_log_message(
        18,
        "[TCN Margin 0.08 < min 0.12]",
        linear=0,
        pending_loss=0.0,
    )
    assert "C0018" in message
    assert "Motivo:" in message
    assert "linear=0" in message
    assert "pending_loss=$0.00" in message
    assert "TCN Margin" in message


def test_quality_conviction_suspends_cluster_ignores_non_dict_decisions(orch_ready):
    assert quality_conviction_suspends_cluster(orch_ready, []) is False


def test_quality_conviction_suspends_cluster_skips_malformed_entries(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 2
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    decisions = {
        "RDBULL": "invalid",
        "RDBEAR": {"metrics": "invalid"},
        "RDBULL2": {
            "metrics": {
                "calibrated_prob": 0.51,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
                "meta_payoff_edge_zscore": 0.10,
                "edge_zscore_samples": 15,
                "deploy_ok": True,
                "direction": "CALL",
            }
        },
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is True
    assert decisions["RDBULL2"]["metrics"]["quality_guard_reject"] is True
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert len(guard_logs) == 1
    assert "C0002" in guard_logs[0].message
    assert "TCN Margin" in guard_logs[0].message or "Meta Z-Score" in guard_logs[0].message
    assert "linear=0" in guard_logs[0].message


def test_quality_conviction_waives_suspension_during_recovery_mandatory(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 7
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.pending_loss_total = lambda: 4.14
    decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.61,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
            }
        },
        "RDBEAR": {"metrics": {"calibrated_prob": 0.30, "predicted_payoff_edge": 0.08}},
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_quality_conviction_suspends_cluster_false_for_regular_elastic_signal(orch_ready):
    orch = orch_ready
    decisions = {
        "RDBULL": {"metrics": _edge_signal_metrics()},
        "RDBEAR": {"metrics": {"calibrated_prob": 0.30, "predicted_payoff_edge": 0.06}},
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_log_quality_guard_suspension_deduplicates_per_cycle(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 3
    reason = "[TCN Margin 0.08 < min 0.12]"
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch, reason=reason)
        log_quality_guard_suspension(orch, reason=reason)
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert len(guard_logs) == 1
    assert "C0003" in guard_logs[0].message
    assert reason in guard_logs[0].message
    assert "min" in guard_logs[0].message


def test_log_quality_guard_suspension_uses_default_reason_when_missing(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 5
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch)
    guard_logs = [
        record for record in caplog.records if "QUALITY_GUARD" in record.message or "EXECUTION_FLOW" in record.message
    ]
    assert len(guard_logs) == 1
    assert "suspenso por meta-regressor" in guard_logs[0].message
