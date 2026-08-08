from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import recovery_gating_active
from src.application.services.deep_learning.dl_params import (
    bars_per_day,
    parse_dl_params,
    resolve_training_history_bars,
    resolve_validation_bars,
    slice_dl_ohlc_window,
    slice_dl_price_window,
)
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    MarketDirectionClassifier,
    fit_norm_stats,
)
from src.domain.models.trade import TradeDirection


def test_bars_per_day_and_training_history_window():
    assert bars_per_day(60) == 1440
    assert resolve_training_history_bars({}, {"granularity": 60}) == 129600
    assert resolve_training_history_bars({"training_history_bars": 120}, {}) == 120
    ratio_val = resolve_validation_bars(
        {"validation_ratio": 0.15},
        training_history_bars=130000,
        lookback=30,
        label_horizon_bars=1,
        label_smooth_bars=1,
    )
    assert ratio_val == 19495
    prices = np.arange(400, dtype=np.float64)
    trimmed = slice_dl_price_window(prices, training_history_bars=288)
    assert len(trimmed) == 288
    assert float(trimmed[0]) == 112.0
    short = np.arange(50, dtype=np.float64)
    kept = slice_dl_price_window(short, training_history_bars=288)
    assert len(kept) == 50
    ohlc_prices = np.arange(400, dtype=np.float64)
    ohlc_open = ohlc_prices - 0.1
    ohlc_high = ohlc_prices + 0.2
    ohlc_low = ohlc_prices - 0.2
    p2, o2, h2, l2 = slice_dl_ohlc_window(
        ohlc_prices,
        training_history_bars=288,
        open_=ohlc_open,
        high=ohlc_high,
        low=ohlc_low,
    )
    assert len(p2) == 288
    assert len(o2) == len(h2) == len(l2) == 288


def test_parse_dl_params():
    full = parse_dl_params(
        {"lookback": 30, "validation_ratio": 0.15, "training_history_bars": 130000},
        {"granularity": 60},
        {"duration": 60, "duration_unit": "s"},
    )
    assert full["training_history_bars"] == 130000
    assert full["validation_bars"] == 19495
    assert full["epochs"] == 50
    assert full["early_stopping_patience"] == 6
    assert full["label_horizon_bars"] == 1
    assert full["confidence_call_threshold"] == 0.75
    assert full["confidence_put_threshold"] == 0.25
    assert full["inference_history_bars"] < full["training_history_bars"]
    assert full["inference_history_bars"] >= full["lookback"] + 16
    explicit = parse_dl_params(
        {"lookback": 30, "training_history_bars": 500, "inference_history_bars": 80},
        {"granularity": 900},
        {},
    )
    assert explicit["inference_history_bars"] == 80
    tcn = parse_dl_params({"tcn": {"channels": [64, 32, 16], "dropout": 0.2}})
    assert tcn["tcn_channels"] == (64, 32, 16)
    assert tcn["tcn_dropout"] == 0.2


def test_recovery_gating_active_when_pending_loss():
    orch = MagicMock()
    orch.risk_manager.pending_loss = {"R_10": 100.92}
    orch.risk_manager.consecutive_losses_linear = 0
    assert recovery_gating_active(orch) is True
    orch.risk_manager.pending_loss = {}
    assert recovery_gating_active(orch) is False
    orch.risk_manager.pending_loss = {"R_10": "bad"}
    assert recovery_gating_active(orch) is False
    orch.risk_manager = None
    assert recovery_gating_active(orch) is False


def test_resolve_dl_model_path_legacy():
    path = resolve_dl_model_path({"model_path": "data/legacy_model.pth"}, "X")
    assert path.name == "legacy_model.pth"
    templated = resolve_dl_model_path({"model_path_template": "data/dl/{symbol}.pth"}, "R_10")
    assert templated.name == "R_10.pth"


def test_predict_symbol_decision_executes_on_confidence():
    params = parse_dl_params(
        {
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
        }
    )
    runtime = {
        "val_accuracy": 0.55,
        "val_brier": 0.2,
        "val_ece": 0.1,
        "lookback": 15,
    }
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.80, 0.80),
    ):
        orch = type("O", (), {"config": {"deep_learning": {}}})()
        entry = predict_symbol_decision(
            orch,
            "R_10",
            MarketDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["trade_score"] == 0.80
    assert entry["metrics"]["raw_prob"] == 0.80
