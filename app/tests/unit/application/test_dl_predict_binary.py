from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import INPUT_DIM, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_predict_abstains_on_low_confidence():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "confidence"
    assert entry["direction"] == TradeDirection.CALL
    assert entry["metrics"]["trade_score"] == pytest.approx(0.52, abs=1e-6)


def test_predict_executes_on_strong_call():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.80, 0.80),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
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


def test_predict_weak_direction_on_neutral_zone():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.57,
            "confidence_put_threshold": 0.43,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.47, 0.47),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_100",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.PUT
    assert entry["metrics"]["trade_score"] == pytest.approx(0.53, abs=1e-6)
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "confidence"


def test_predict_mandatory_trend_fallback():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    orch = type("O", (), {"config": {"orchestrator": {"execution": {"mandatory_trade_each_cycle": True}}}})()
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}

    # Test CALL trend fallback (last price > average)
    prices_call = np.array([10.0] * 79 + [12.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_call,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.CALL
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["trend_fallback"] is True

    # Test PUT trend fallback (last price < average)
    prices_put = np.array([10.0] * 79 + [8.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_put,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.PUT
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["trend_fallback"] is True


def test_predict_dynamic_trend_ema_vs_sma():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}

    # Caso 1: EMA com trend_period = 3
    orch_ema = type(
        "O",
        (),
        {
            "config": {
                "orchestrator": {
                    "execution": {
                        "mandatory_trade_each_cycle": True,
                        "trend_period": 3,
                        "trend_use_ema": True,
                    }
                }
            }
        },
    )()
    # Precos onde a EMA-3 vai dar CALL (recente subindo rapido)
    prices = np.array([10.0, 5.0, 6.0, 9.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_ema,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "CALL"

    # Caso 2: SMA com trend_period = 3
    orch_sma = type(
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
    # Precos onde a SMA-3 vai dar PUT (media dos ultimos 3: (5+6+1)/3 = 4, preco atual 3 < 4)
    prices_sma = np.array([10.0, 5.0, 6.0, 3.0])
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_sma,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_sma,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "PUT"

    # Caso 3: prices vazio (t_len <= 0)
    prices_empty = np.array([], dtype=np.float64)
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.52, 0.52),
    ):
        entry = predict_symbol_decision(
            orch_sma,
            "R_50",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            prices_empty,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["trend_direction"] == "CALL"
