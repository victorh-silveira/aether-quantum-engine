from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_predict_abstains_on_gray_zone_raw_prob():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
            "calibration": {"neutral_half_width": 0.02, "calibration_neutral_drift": [0.48, 0.52]},
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}, "orchestrator": {"execution": {}}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(None, 0.50, 0.50),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.CALL
    assert entry["metrics"].get("calibration_mode") == "calibrated"
    assert entry["metrics"]["calibrated_prob"] == pytest.approx(0.50)


def test_predict_executes_on_strong_call():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}, "orchestrator": {"execution": {}}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.80, 0.80),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["direction"] == TradeDirection.CALL


def test_predict_weak_direction_still_executes():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.57,
            "confidence_put_threshold": 0.43,
            "min_val_accuracy": 0.53,
            "calibration": {"neutral_half_width": 0.04},
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}, "orchestrator": {"execution": {}}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(None, 0.42, 0.42),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.PUT
    assert entry["metrics"]["trade_score"] == pytest.approx(0.58, abs=1e-6)
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["gate_reason"] is None


def test_predict_includes_dynamic_threshold_metrics():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.53,
            "confidence_put_threshold": 0.47,
            "min_edge_execute": 0.04,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "dynamic_threshold": {
                            "enabled": True,
                            "vol_source": "blend",
                            "baseline_lookback": 8,
                        }
                    }
                }
            }
        },
    )()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.80, 0.80),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.linspace(10.0, 11.0, 80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["calibrated_prob"] == 0.80
    assert "dynamic_call_threshold" in entry["metrics"]
    assert "volatility_regime" in entry["metrics"]


def test_predict_includes_trend_metrics():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "mandatory_trade_each_cycle": True,
                        "trend_period": 3,
                        "trend_use_ema": False,
                    }
                }
            }
        },
    )()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    prices = np.array([10.0, 5.0, 6.0, 3.0])
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(None, 0.60, 0.60),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "PUT"
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["calibration_mode"] == "calibrated"


def test_predict_trend_conflict_does_not_block():
    params = parse_dl_params({"confidence_call_threshold": 0.75, "confidence_put_threshold": 0.25})
    orch = type(
        "O",
        (),
        {"config": {"orchestrator": {"execution": {"trend_period": 3, "trend_use_ema": False}}}},
    )()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    prices = np.array([5.0, 5.0, 6.0, 9.0])
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(None, 0.45, 0.45),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["gate_reason"] is None
    assert entry["direction"] == TradeDirection.PUT
