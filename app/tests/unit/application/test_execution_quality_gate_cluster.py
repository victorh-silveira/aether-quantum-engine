from datetime import UTC, datetime
from types import SimpleNamespace

from src.application.services.execution_quality_gate_cluster import (
    log_quality_guard_suspension,
    quality_conviction_suspends_cluster,
)


def _edge_signal_metrics() -> dict:
    return {
        "calibrated_prob": 0.57,
        "predicted_payoff_edge": 0.06,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.55,
    }


def _weak_edge_metrics() -> dict:
    return {
        "calibrated_prob": 0.61,
        "predicted_payoff_edge": 0.01,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.10,
        "deploy_ok": True,
        "direction": "CALL",
    }


def test_quality_conviction_suspends_cluster_ignores_non_dict_decisions(orch_ready):
    assert quality_conviction_suspends_cluster(orch_ready, []) is False


def test_quality_conviction_suspends_cluster_skips_malformed_entries(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 2
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.dlambert_unit = 16.0
    orch.risk_manager.pending_loss_total = lambda: 20.0
    decisions = {
        "RDBULL": "invalid",
        "RDBEAR": {"metrics": "invalid"},
        "RDBULL2": {"metrics": _weak_edge_metrics()},
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is True
    guard_logs = [record for record in caplog.records if "EXECUTION_FLOW" in record.message]
    assert len(guard_logs) == 1


def test_quality_conviction_suspends_cluster_keeps_decisions_unblocked_in_recovery(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 7
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.dlambert_unit = 16.0
    orch.risk_manager.pending_loss_total = lambda: 20.0
    decisions = {
        "RDBULL": {"metrics": _weak_edge_metrics()},
        "RDBEAR": {
            "metrics": {
                "calibrated_prob": 0.30,
                "predicted_payoff_edge": 0.08,
                "meta_payoff_edge_zscore": 0.55,
                "deploy_ok": True,
                "direction": "PUT",
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert decisions["RDBULL"]["metrics"].get("execution_gate_state") == "meta_zscore_reject"
    assert decisions["RDBEAR"]["metrics"].get("execution_gate_state") == "meta_zscore_pass"


def test_quality_conviction_suspends_cluster_false_for_regular_elastic_signal(orch_ready):
    orch = orch_ready
    decisions = {
        "RDBULL": {"metrics": _edge_signal_metrics() | {"deploy_ok": True, "direction": "CALL"}},
        "RDBEAR": {
            "metrics": {
                "calibrated_prob": 0.30,
                "predicted_payoff_edge": 0.06,
                "meta_payoff_edge_zscore": 0.55,
                "deploy_ok": True,
                "direction": "PUT",
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_log_quality_guard_suspension_noop_without_logger():
    orch = SimpleNamespace(logger=None, _active_cycle_id=1, risk_manager=None)
    log_quality_guard_suspension(orch, reason="[Meta Z-Score]")


def test_log_quality_guard_suspension_deduplicates_within_same_minute(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 44
    orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 1, tzinfo=UTC)
    reason = "[TCN Margin 0.11 < min 0.12]"
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch, reason=reason)
        orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 17, tzinfo=UTC)
        log_quality_guard_suspension(orch, reason="[TCN Margin 0.10 < min 0.12]")
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert len(guard_logs) == 1
    assert "C0044" in guard_logs[0].message


def test_quality_conviction_suspends_cluster_suppresses_sub_minute_duplicate_logs(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 44
    orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 1, tzinfo=UTC)
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.dlambert_unit = 16.0
    orch.risk_manager.pending_loss_total = lambda: 20.0
    decisions = {"RDBULL": {"metrics": _weak_edge_metrics()}}
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is True
        orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 17, tzinfo=UTC)
        assert quality_conviction_suspends_cluster(orch, decisions) is True
    guard_logs = [record for record in caplog.records if "EXECUTION_FLOW" in record.message]
    assert len(guard_logs) == 1


def test_log_quality_guard_suspension_emits_again_on_next_minute_bucket(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 44
    orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 59, tzinfo=UTC)
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch, reason="[TCN Margin 0.11 < min 0.12]")
        orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 11, 2, tzinfo=UTC)
        log_quality_guard_suspension(orch, reason="[TCN Margin 0.11 < min 0.12]")
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert len(guard_logs) == 2


def test_quality_conviction_suspends_cluster_skips_deploy_blocked_entries(orch_ready):
    orch = orch_ready
    decisions = {"RDBULL": {"direction": "CALL", "metrics": {"deploy_ok": False, "calibrated_prob": 0.7}}}
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_log_quality_guard_suspension_uses_default_reason_when_missing(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 5
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch)
    guard_logs = [record for record in caplog.records if "EXECUTION_FLOW" in record.message]
    assert len(guard_logs) == 1
    assert "suspenso por meta-regressor" in guard_logs[0].message
