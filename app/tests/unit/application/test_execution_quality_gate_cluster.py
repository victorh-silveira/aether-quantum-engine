from datetime import UTC, datetime
from types import SimpleNamespace

from src.application.services.execution_quality_gate_cluster import (
    _mandatory_trade_each_cycle,
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
        "calibrated_prob": 0.51,
        "predicted_payoff_edge": 0.01,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.10,
        "edge_zscore_samples": 15,
        "deploy_ok": True,
        "direction": "CALL",
    }


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
        "RDBULL2": {"metrics": _weak_edge_metrics()},
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_quality_conviction_mandatory_flat_logs_without_suspending(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 5
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    decisions = {"RDBULL": {"metrics": {"calibrated_prob": 0.51, "deploy_ok": True}}}
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert orch._last_quality_gate_regime == "mandatory_continuous"


def test_quality_conviction_mandatory_continues_on_negative_edge_without_strong_z(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 27
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.pending_loss_total = lambda: 140.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    decisions = {
        "RDBULL": {
            "metrics": {
                "deploy_ok": True,
                "direction": "CALL",
                "raw_prob": 0.64,
                "predicted_payoff_edge": -0.15,
                "meta_payoff_edge_zscore": 0.10,
                "edge_zscore_samples": 15,
                "calibrated_prob": 0.64,
            }
        }
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert getattr(orch, "_last_quality_gate_regime", None) != "mandatory_meta_hard_skip"


def test_quality_conviction_mandatory_hard_skips_strongly_negative_meta(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 26
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    decisions = {
        "RDBULL": {
            "metrics": {
                "deploy_ok": True,
                "direction": "PUT",
                "raw_prob": 0.40,
                "predicted_payoff_edge": -2.02,
                "meta_payoff_edge_zscore": -3.09,
                "edge_zscore_samples": 15,
                "calibrated_prob": 0.51,
            }
        },
        "RDBEAR": {
            "metrics": {
                "deploy_ok": True,
                "direction": "PUT",
                "raw_prob": 0.45,
                "predicted_payoff_edge": -0.56,
                "meta_payoff_edge_zscore": -0.80,
                "edge_zscore_samples": 15,
                "calibrated_prob": 0.51,
            }
        },
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert orch._last_quality_gate_regime == "mandatory_continuous"
    assert decisions["RDBULL"]["metrics"].get("signal_status") != "SIGNAL_SUSPENDED"


def test_quality_conviction_mandatory_continues_on_emergency_waiver(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 40
    orch.risk_manager.consecutive_losses_linear = 5
    orch.risk_manager.pending_loss = {"RDBULL": 260.0}
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    decisions = {
        "bad": "skip",
        "RDBEAR": {"metrics": "invalid"},
        "RDBULL": {
            "metrics": {
                "deploy_ok": True,
                "direction": "PUT",
                "raw_prob": 0.18,
                "predicted_payoff_edge": -1.20,
                "meta_payoff_edge_zscore": -1.50,
                "edge_zscore_samples": 15,
                "calibrated_prob": 0.50,
                "meta_classifier_applied": True,
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert orch._last_quality_gate_regime == "mandatory_continuous"


def test_quality_conviction_suspends_cluster_keeps_decisions_unblocked_with_meta_pass(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 7
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
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
    assert decisions["RDBULL"]["metrics"].get("execution_gate_state") == "meta_payoff_gate_disabled"
    assert decisions["RDBEAR"]["metrics"].get("execution_gate_state") == "meta_payoff_gate_disabled"


def test_sniper_cluster_keeps_strong_tcn_when_peer_neutral_clamp(orch_ready):
    orch = orch_ready
    orch._active_cycle_id = 1
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    decisions = {
        "RDBEAR": {
            "metrics": {
                "deploy_ok": True,
                "direction": None,
                "calibrated_prob": 0.50,
                "calibration_mode": "neutral_clamp",
                "gate_reason": "neutral_clamp",
                "execute": False,
            }
        },
        "RDBULL": {
            "metrics": {
                "deploy_ok": True,
                "direction": "CALL",
                "calibrated_prob": 0.84,
                "direction_margin": 0.34,
                "execute": True,
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False
    assert decisions["RDBULL"]["metrics"].get("signal_status") != "SIGNAL_SUSPENDED"


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
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    decisions = {"RDBULL": {"metrics": _weak_edge_metrics()}}
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions) is False
        orch._broker_server_time_utc = datetime(2026, 7, 7, 23, 10, 17, tzinfo=UTC)
        assert quality_conviction_suspends_cluster(orch, decisions) is False
    guard_logs = [record for record in caplog.records if "QUALITY_GUARD" in record.message]
    assert len(guard_logs) == 0


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


def test_quality_conviction_waives_suspension_during_recovery_mandatory(orch_ready):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.pending_loss_total = lambda: 4.14
    decisions = {"RDBULL": {"metrics": _weak_edge_metrics()}}
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_quality_conviction_suspends_cluster_skips_deploy_blocked_entries(orch_ready):
    orch = orch_ready
    decisions = {"RDBULL": {"direction": "CALL", "metrics": {"deploy_ok": False, "calibrated_prob": 0.7}}}
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_mandatory_trade_each_cycle_defaults_when_exec_cfg_invalid():
    assert _mandatory_trade_each_cycle(None) is True
    assert _mandatory_trade_each_cycle("invalid") is True


def test_log_quality_guard_suspension_uses_default_reason_when_missing(orch_ready, caplog):
    orch = orch_ready
    orch._active_cycle_id = 5
    with caplog.at_level("INFO", logger="AETH"):
        log_quality_guard_suspension(orch)
    guard_logs = [
        record for record in caplog.records if "QUALITY_GUARD" in record.message or "EXECUTION_FLOW" in record.message
    ]
    assert len(guard_logs) == 1
    assert "suspenso por quality gate" in guard_logs[0].message
