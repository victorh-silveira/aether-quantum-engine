from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_predict_abstains_on_exhaustion_conflict():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.55,
            "confidence_put_threshold": 0.45,
            "min_val_accuracy": 0.50,
            "exhaustion_filter_enabled": True,
            "exhaustion_rsi_lower": 0.28,
            "exhaustion_rsi_upper": 0.72,
        }
    )
    orch = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "trend_period": 3,
                        "trend_use_ema": False,
                    }
                }
            }
        },
    )()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    prices = np.array([5.0, 5.0, 6.0, 9.0])

    # Test case 1: Strong PUT direction, but RSI is extremely oversold (e.g. 0.25)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.PUT, 0.40, 0.40),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict.precompute_price_series",
            return_value={"rsi": [0.25], "adx": [0.15], "hurst": [0.5]},
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "exhaustion_conflict"

    # Test case 2: Strong CALL direction, but Keltner pct b is overbought (e.g. 1.2)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.70, 0.70),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict.precompute_price_series",
            return_value={"rsi": [0.50], "keltner_pct_b": [1.2], "adx": [0.15], "hurst": [0.5]},
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "exhaustion_conflict"

    # Test case 3: Neutral direction (None), weak direction PUT, but RSI is extremely oversold (e.g. 0.25)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(None, 0.40, 0.40),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict.precompute_price_series",
            return_value={"rsi": [0.25], "adx": [0.15], "hurst": [0.5]},
        ),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "exhaustion_conflict"
