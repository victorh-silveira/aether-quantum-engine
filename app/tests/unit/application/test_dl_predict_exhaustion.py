from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_predict_exhaustion_does_not_block_execution():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.55,
            "confidence_put_threshold": 0.45,
            "min_val_accuracy": 0.50,
        }
    )
    orch = type(
        "O",
        (),
        {"config": {"orchestrator": {"execution": {"trend_period": 3, "trend_use_ema": False}}}},
    )()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15, "deploy_ok": True}
    prices = np.array([5.0, 5.0, 6.0, 9.0])

    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.PUT, 0.40, 0.40),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_build.precompute_price_series",
            return_value={
                "rsi": [0.25],
                "adx": [0.15],
                "hurst": [0.5],
                "vol_ratio_short_long": [1.0],
                "cmo": [0.0],
                "keltner_pct_b": [0.5],
                "macd": [0.0],
                "macd_signal": [0.0],
                "di_diff": [0.0],
            },
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "RDBULL",
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
    assert "rsi" in entry["metrics"]["indicators"]
