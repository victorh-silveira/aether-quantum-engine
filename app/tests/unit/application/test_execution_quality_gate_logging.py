from src.application.services.execution_quality_gate_cluster import (
    log_quality_guard_suspension,
)


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
