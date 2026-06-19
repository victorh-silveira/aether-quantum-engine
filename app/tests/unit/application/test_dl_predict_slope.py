from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats


def test_predict_trend_slope():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}

    # Caso 1: Slope com SMA, trend_period = 3, trend_use_slope = True
    orch_slope_sma = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "mandatory_trade_each_cycle": True,
                        "trend_period": 3,
                        "trend_use_ema": False,
                        "trend_use_slope": True,
                    }
                }
            }
        },
    )()

    # Precos onde SMA atual é maior que a anterior (inclinação positiva)
    # close_prices = [10.0, 11.0, 12.0] -> SMA atual de [10.0, 11.0, 12.0] = 11.0
    # prev_prices = [10.0, 11.0] -> SMA anterior de [10.0, 11.0] = 10.5
    # Como 11.0 >= 10.5, deve retornar CALL
    prices_up = np.array([10.0, 11.0, 12.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_slope_sma,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_up,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "CALL"

    # Precos onde SMA atual é menor que a anterior (inclinação negativa)
    # close_prices = [12.0, 11.0, 10.0] -> SMA atual = 11.0
    # prev_prices = [12.0, 11.0] -> SMA anterior = 11.5
    # Como 11.0 < 11.5, deve retornar PUT
    prices_down = np.array([12.0, 11.0, 10.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_slope_sma,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_down,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "PUT"

    # Caso 2: Slope com EMA
    orch_slope_ema = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "mandatory_trade_each_cycle": True,
                        "trend_period": 3,
                        "trend_use_ema": True,
                        "trend_use_slope": True,
                    }
                }
            }
        },
    )()
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_slope_ema,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_up,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "CALL"
