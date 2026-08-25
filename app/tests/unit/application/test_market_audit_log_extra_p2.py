"""CLUSTER audit line extras (parte 2)."""

from src.application.services.market_audit_log import format_cluster_audit_line


def test_format_cluster_neg_edge_shows_raw_edge_and_be():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "raw_prob": 0.99,
                "calibrated_prob": 0.533,
                "exec_direction": "CALL",
                "gate_reason": "neg_edge",
                "signal_status": "SKIP:NEG_EDGE",
            },
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "raw_edge:" in line and "be=0.541" in line and "Edge: -0.014" in line
    assert "p_call: 0.53300" in line and "p_put: 0.46700" in line
    assert "NEG_EDGE" in line or "neg_edge" in line


def test_format_cluster_edge_gap_before_gate():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {"raw_prob": 0.98783, "calibrated_prob": 0.53338, "exec_direction": "CALL"},
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "Edge: -0.013" in line and "raw_edge: +0.827" in line and "be=0.541" in line
    assert "Margin: 0.033" in line and "p_call: 0.53338" in line and "p_put: 0.46662" in line


def test_format_cluster_leans_call_when_direction_missing():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {"raw_prob": 0.37115, "calibrated_prob": 0.52497},
        }
    }
    line = format_cluster_audit_line(decisions, timeframe="M2")
    assert "R_10: CALL (" in line and "p_call: 0.52497" in line and "p_put: 0.47503" in line
