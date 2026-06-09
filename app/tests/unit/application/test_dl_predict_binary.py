from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_predict_binary_noise_floor():
    params = parse_dl_params(
        {
            "min_conviction_execute": 0.50,
            "min_edge_margin": 0.01,
            "min_val_accuracy": 0.0,
            "require_regime_alignment": False,
            "min_direction_margin": 0.01,
            "binary_signal": {"min_rel_vol_execute": 0.99},
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.7, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    flat = np.full(80, 100.0, dtype=np.float64)
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.72, 0.72, 0.72),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            flat,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "noise_floor"


def test_predict_stat_override_flag():
    params = parse_dl_params(
        {
            "min_conviction_execute": 0.50,
            "min_edge_margin": 0.01,
            "min_val_accuracy": 0.0,
            "require_regime_alignment": False,
            "min_direction_margin": 0.01,
            "binary_signal": {
                "sma_z_extreme": 0.0001,
                "weak_dl_override_margin": 0.30,
                "min_rel_vol_execute": 0.0,
            },
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.7, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    up = np.linspace(100.0, 200.0, 80, dtype=np.float64)
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.52, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            up,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"].get("stat_override") is True
    assert entry["direction"] == TradeDirection.PUT
