import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_HARMONIC_WINDOW,
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.execution_direction_resolver import (
    D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection


def _entry(*, calibrated_prob=0.70):
    return {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "calibrated_prob": calibrated_prob,
            "val_accuracy": 0.70,
        },
    }


def test_resolve_soft_bb_compression_keeps_organic_score_at_fifty_five_percent_ratio():
    reset_bb_width_buffer()
    for _ in range(BB_WIDTH_HARMONIC_WINDOW):
        record_bb_width(0.068)
    entry = _entry()
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.041}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.02}
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("meta_squeeze_downgrade") is not True
    assert metrics["trade_score"] == pytest.approx(0.70)
    assert metrics["bb_width_anomaly_ratio"] == pytest.approx(D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO)
    assert (0.041 / metrics["bb_width_harmonic_mean"]) + 1e-12 >= D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO


def test_resolve_frozen_book_compression_triggers_d_squeeze_at_fifty_five_percent_ratio():
    reset_bb_width_buffer()
    for _ in range(BB_WIDTH_HARMONIC_WINDOW):
        record_bb_width(0.068)
    entry = _entry()
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.030}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.02}
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["meta_squeeze_downgrade"] is True
    assert metrics["trade_score"] == pytest.approx(0.52)
    assert (0.030 / metrics["bb_width_harmonic_mean"]) + 1e-12 < D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO
