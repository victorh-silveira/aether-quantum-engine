from types import SimpleNamespace

import pytest

from src.application.services.execution_quality_gate_meta import (
    _meta_payoff_zscore,
    emit_quality_reject_log,
    ensure_meta_zscore_telemetry,
    evaluate_meta_payoff_quality,
    meta_zscore_reject_reason,
    resolve_min_meta_payoff_zscore,
)


def test_meta_payoff_zscore_reads_edge_zscore_when_meta_field_missing():
    assert _meta_payoff_zscore({"edge_zscore": 0.44}) == pytest.approx(0.44)


def test_resolve_min_meta_payoff_zscore_reads_config():
    assert resolve_min_meta_payoff_zscore({"quality_gate": {"min_meta_payoff_zscore": 0.60}}) == 0.60


def test_meta_zscore_reject_reason_formats_below_min():
    assert "< min 0.50" in meta_zscore_reject_reason(0.35, min_z=0.5)


def test_evaluate_meta_payoff_quality_accepts_favorable_zscore():
    metrics = {
        "calibrated_prob": 0.60,
        "predicted_payoff_edge": 0.06,
        "meta_payoff_edge_zscore": 0.55,
    }
    assert evaluate_meta_payoff_quality(metrics, exec_cfg={}) is True
    assert metrics["execution_gate_state"] == "meta_zscore_pass"


def test_evaluate_meta_payoff_quality_rejects_low_zscore():
    metrics = {
        "calibrated_prob": 0.57,
        "predicted_payoff_edge": 0.02,
        "meta_payoff_edge_zscore": 0.10,
    }
    assert evaluate_meta_payoff_quality(metrics, exec_cfg={}) is False
    assert "< min 0.50" in metrics["quality_gate_reason"]


def test_evaluate_meta_payoff_quality_rejects_negative_edge():
    metrics = {
        "predicted_payoff_edge": -0.01,
        "meta_payoff_edge_zscore": -0.20,
    }
    assert evaluate_meta_payoff_quality(metrics, exec_cfg={}) is False


def test_evaluate_meta_payoff_quality_rejects_below_min_z():
    metrics = {
        "predicted_payoff_edge": 0.04,
        "meta_payoff_edge_zscore": 0.35,
    }
    assert evaluate_meta_payoff_quality(metrics, exec_cfg={}) is False
    assert "< min 0.50" in metrics["quality_gate_reason"]


def test_meta_payoff_zscore_reads_edge_zscore_fallback():
    metrics = {"edge_zscore": 0.62, "predicted_payoff_edge": 0.05}
    assert evaluate_meta_payoff_quality(metrics, exec_cfg={}) is True


def test_evaluate_meta_payoff_quality_logs_reject_when_requested(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 12
    metrics = {
        "predicted_payoff_edge": 0.01,
        "meta_payoff_edge_zscore": 0.05,
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert (
            evaluate_meta_payoff_quality(
                metrics,
                exec_cfg={},
                orch=orch,
                log_reject=True,
                minute_bucket="202607072310",
            )
            is False
        )
    guard_logs = [record for record in caplog.records if "EXECUTION_FLOW" in record.message]
    assert len(guard_logs) == 1


def test_ensure_meta_zscore_telemetry_populates_missing_fields():
    metrics = {"predicted_payoff_edge": 0.04}
    ensure_meta_zscore_telemetry(metrics, linear=0, pending_loss_total=0.0)
    assert "meta_payoff_edge_zscore" in metrics


def test_emit_quality_reject_log_uses_log_deduper(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 9
    with caplog.at_level("INFO", logger="AETH"):
        emit_quality_reject_log(
            orch,
            cycle_id=9,
            reason="[Meta Z-Score 0.10 < min 0.50]",
            minute_bucket="202607072310",
        )
    guard_logs = [record for record in caplog.records if "EXECUTION_FLOW" in record.message]
    assert len(guard_logs) == 1
    assert "suspenso por meta-regressor" in guard_logs[0].message


def test_ensure_meta_zscore_telemetry_skips_when_edge_zscore_present():
    metrics = {"predicted_payoff_edge": 0.04, "edge_zscore": 0.71}
    ensure_meta_zscore_telemetry(metrics, linear=0, pending_loss_total=0.0)
    assert metrics["edge_zscore"] == pytest.approx(0.71)
    assert "meta_payoff_edge_zscore" not in metrics


def test_emit_quality_reject_log_without_logger_is_noop():
    orch = SimpleNamespace(logger=None)
    emit_quality_reject_log(orch, cycle_id=1, reason="x", minute_bucket="m")
