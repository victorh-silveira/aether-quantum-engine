from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from tests.unit.application.test_execution_quality_gate import _edge_signal_metrics


def test_quality_conviction_suspends_cluster_false_for_regular_elastic_signal(orch_ready):
    orch = orch_ready
    decisions = {
        "R_10": {"metrics": _edge_signal_metrics()},
        "R_50": {"metrics": {"calibrated_prob": 0.30, "predicted_payoff_edge": 0.06}},
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False
