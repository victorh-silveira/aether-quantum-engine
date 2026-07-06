import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_HARMONIC_WINDOW,
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.meta_direction_flip import micro_volatility_squeeze_active
from src.application.services.meta_payoff_regression import (
    META_SQUEEZE_TRADE_SCORE,
    apply_meta_regression_edge,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_bb_buffer():
    reset_bb_width_buffer()
    yield
    reset_bb_width_buffer()


def _prime_bb(value: float, count: int = BB_WIDTH_HARMONIC_WINDOW) -> None:
    for _ in range(count):
        record_bb_width(value)


def test_micro_volatility_squeeze_active_bb_width():
    _prime_bb(0.050)
    metrics = {"indicators": {"bb_width": 0.020}}
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


def test_apply_meta_regression_edge_loss_expected_triggers_squeeze(caplog):
    metrics = {
        "indicators": {"bb_width": 0.12},
        "flow_features": {"micro_tick_acceleration": 0.03},
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


def test_apply_meta_regression_edge_bb_compression_triggers_squeeze_even_positive_edge(caplog):
    _prime_bb(0.050)
    metrics = {
        "indicators": {"bb_width": 0.020},
        "flow_features": {"micro_tick_acceleration": 0.05},
    }
    with caplog.at_level("INFO"):
        direction, score = apply_meta_regression_edge(
            TradeDirection.CALL,
            metrics,
            0.18,
            meta_applied=True,
            base_score=0.72,
            symbol="RDBULL",
        )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(META_SQUEEZE_TRADE_SCORE)
    assert metrics["meta_squeeze_downgrade"] is True
    assert any("[D-SQUEEZE]" in record.message for record in caplog.records)


def test_apply_meta_regression_edge_mild_negative_triggers_squeeze():
    metrics = {"indicators": {"bb_width": 0.12}, "flow_features": {"micro_tick_acceleration": 0.03}}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.05,
        meta_applied=True,
        base_score=0.71,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(META_SQUEEZE_TRADE_SCORE)
    assert metrics["meta_squeeze_downgrade"] is True


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
    assert metrics.get("meta_squeeze_downgrade") is not True


def test_meta_squeeze_trade_score_constant():
    assert pytest.approx(0.52) == META_SQUEEZE_TRADE_SCORE
