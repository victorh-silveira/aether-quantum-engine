import pytest

from src.application.services.meta_direction_flip import micro_volatility_squeeze_active
from src.application.services.meta_payoff_regression import (
    META_SEVERE_NEGATIVE_EDGE,
    META_SQUEEZE_TRADE_SCORE,
    apply_meta_regression_edge,
)
from src.domain.models.trade import TradeDirection


def test_micro_volatility_squeeze_active_bb_width():
    metrics = {"indicators": {"bb_width": 0.04}}
    assert micro_volatility_squeeze_active(metrics) is True


def test_micro_volatility_squeeze_active_tick_deceleration():
    metrics = {"flow_features": {"micro_tick_acceleration": -0.01}}
    assert micro_volatility_squeeze_active(metrics) is True


def test_apply_meta_regression_edge_positive_keeps_organic_score():
    metrics = {}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.12,
        meta_applied=True,
        base_score=0.74,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.74)
    assert metrics["predicted_payoff_edge"] == pytest.approx(0.12)


def test_apply_meta_regression_edge_severe_negative_squeeze_downgrades(caplog):
    metrics = {
        "indicators": {"bb_width": 0.03},
        "flow_features": {"micro_tick_acceleration": -0.02},
    }
    with caplog.at_level("INFO"):
        direction, score = apply_meta_regression_edge(
            TradeDirection.PUT,
            metrics,
            -0.25,
            meta_applied=True,
            base_score=0.68,
            symbol="RDBEAR",
        )
    assert direction == TradeDirection.PUT
    assert score == pytest.approx(META_SQUEEZE_TRADE_SCORE)
    assert metrics["meta_squeeze_downgrade"] is True
    assert any("[D-SQUEEZE]" in record.message for record in caplog.records)


def test_apply_meta_regression_edge_mild_negative_without_squeeze_keeps_organic():
    metrics = {"indicators": {"bb_width": 0.12}, "flow_features": {"micro_tick_acceleration": 0.03}}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.05,
        meta_applied=True,
        base_score=0.71,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.71)


def test_apply_meta_regression_edge_not_applied_uses_base_score():
    metrics = {}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.30,
        meta_applied=False,
        base_score=0.66,
    )
    assert score == pytest.approx(0.66)


def test_meta_severe_negative_edge_constant():
    assert pytest.approx(-0.15) == META_SEVERE_NEGATIVE_EDGE
