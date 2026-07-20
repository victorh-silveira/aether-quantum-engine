import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_payoff_regression import (
    CALIBRATION_NEUTRAL_DRIFT,
    apply_meta_regression_edge,
    calibration_neutral_axis_drift,
    veto_calibration_neutral_drift,
)
from src.domain.models.trade import TradeDirection


def _entry(*, raw_prob: float, calibrated_prob: float, direction: TradeDirection | None = None):
    metrics = {
        "deploy_ok": True,
        "execute": True,
        "raw_prob": raw_prob,
        "calibrated_prob": calibrated_prob,
        "predicted_payoff_edge": 0.08,
        "meta_classifier_applied": True,
        "val_accuracy": 0.65,
    }
    return {"direction": direction, "metrics": metrics}


def test_calibration_neutral_axis_drift_detects_put_to_call_flip():
    assert calibration_neutral_axis_drift(0.42, 0.55) is True


def test_calibration_neutral_axis_drift_detects_call_to_put_flip():
    assert calibration_neutral_axis_drift(0.61, 0.44) is True


def test_calibration_neutral_axis_drift_allows_same_side_compression():
    assert calibration_neutral_axis_drift(0.42, 0.44) is False
    assert calibration_neutral_axis_drift(0.58, 0.62) is False


def test_calibration_neutral_axis_drift_ignores_missing_values():
    assert calibration_neutral_axis_drift(None, 0.55) is False
    assert calibration_neutral_axis_drift(0.42, None) is False


def test_veto_calibration_neutral_drift_annuls_direction_and_trade_score():
    metrics = {"raw_prob": 0.40, "calibrated_prob": 0.53, "trade_score": 0.53}
    assert veto_calibration_neutral_drift(metrics) is False
    assert metrics.get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT
    assert metrics["trade_score"] == 0.53


@pytest.mark.parametrize(
    ("raw_prob", "calibrated_prob"),
    (
        (0.38, 0.55),
        (0.62, 0.45),
    ),
)
def test_resolve_execution_direction_vetoes_absolute_on_neutral_drift(raw_prob, calibrated_prob):
    entry = _entry(raw_prob=raw_prob, calibrated_prob=calibrated_prob)
    result = resolve_execution_direction(entry, symbol="R_10")
    assert result is not None
    assert entry["metrics"].get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT
    assert entry["metrics"].get("resolved_direction") is not None


def test_resolve_execution_direction_allows_aligned_calibration():
    entry = _entry(raw_prob=0.38, calibrated_prob=0.41, direction=TradeDirection.PUT)
    result = resolve_execution_direction(entry, symbol="R_10")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics.get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT


def test_apply_meta_regression_edge_vetoes_on_calibration_drift():
    metrics = {"raw_prob": 0.44, "calibrated_prob": 0.56}
    direction, score = apply_meta_regression_edge(
        TradeDirection.PUT,
        metrics,
        0.10,
        meta_applied=True,
        base_score=0.56,
        symbol="R_10",
    )
    assert direction == TradeDirection.PUT
    assert score > 0.0
    assert metrics.get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT
    assert metrics.get("trade_score") is not None
