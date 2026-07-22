import pytest

from src.application.services.meta_payoff_veto_gate import should_veto_meta_payoff_negative_zscore
from src.domain.models.trade import TradeDirection
from tests.unit.application.test_meta_payoff_veto_gate import _stamp_negative_zscore


def test_soft_veto_deferred_until_zscore_samples_ready():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": -1.20,
        "edge_zscore": -1.20,
        "edge_zscore_samples": 1,
        "edge_zscore_window": 15,
    }
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is False
    assert metrics.get("meta_soft_veto_deferred") is True
    assert metrics["trade_score"] == pytest.approx(0.80)


def test_soft_veto_activates_with_two_samples():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": -1.20,
        "edge_zscore": -1.20,
        "edge_zscore_samples": 2,
        "edge_zscore_window": 15,
    }
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics.get("meta_soft_veto_deferred") is not True


def test_soft_veto_falls_back_to_raw_prob_when_scores_missing():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "raw_prob": 0.81,
    }
    _stamp_negative_zscore(metrics, z_score=-0.90)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["trade_score"] == pytest.approx(0.81 * 0.85)


def test_soft_veto_falls_back_to_resolved_conviction():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "resolved_conviction": 0.77,
        "raw_prob": 0.55,
    }
    _stamp_negative_zscore(metrics, z_score=-0.90)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.PUT) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["trade_score"] == pytest.approx(0.77 * 0.85)
