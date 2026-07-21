import json

import pytest

from aether_paths import repo_path
from src.application.services.bb_width_adaptive_squeeze import (
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _anomaly_ratio() -> float:
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    return float(full["orchestrator"]["execution"]["bb_width_adaptive_squeeze"]["anomaly_ratio"])


def _harmonic_window() -> int:
    return int(load_indicator_config_from_settings()["windows"]["bb_width_harmonic_window"])


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


def test_resolve_soft_bb_compression_keeps_organic_score_at_configured_ratio():
    reset_bb_width_buffer()
    ratio = _anomaly_ratio()
    for _ in range(_harmonic_window()):
        record_bb_width(0.068)
    entry = _entry()
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.041}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.02}
    result = resolve_execution_direction(entry, symbol="R_10")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("meta_squeeze_downgrade") is not True
    assert metrics["trade_score"] == pytest.approx(0.70)
    assert metrics["bb_width_anomaly_ratio"] == pytest.approx(ratio)
    assert (0.041 / metrics["bb_width_harmonic_mean"]) + 1e-12 >= ratio


def test_resolve_frozen_book_compression_triggers_d_squeeze_at_configured_ratio():
    reset_bb_width_buffer()
    ratio = _anomaly_ratio()
    for _ in range(_harmonic_window()):
        record_bb_width(0.068)
    entry = _entry()
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.015}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.02}
    result = resolve_execution_direction(entry, symbol="R_10")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["meta_squeeze_downgrade"] is True
    assert metrics["trade_score"] == pytest.approx(0.52)
    assert (0.015 / metrics["bb_width_harmonic_mean"]) + 1e-12 < ratio
