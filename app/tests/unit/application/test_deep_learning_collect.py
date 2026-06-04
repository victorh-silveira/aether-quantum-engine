from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import recovery_gating_active
from src.application.services.deep_learning.dl_params import (
    bars_per_day,
    optional_float,
    parse_dl_params,
    resolve_training_history_bars,
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
    assert bars_per_day(300) == 288
    assert resolve_training_history_bars({}, {"granularity": 300}) == 288
    assert resolve_training_history_bars({"training_history_bars": 120}, {}) == 120
    assert resolve_training_history_bars({}, {"history_bars": 100}) == 100
    prices = np.arange(400, dtype=np.float64)
    pair = np.arange(400, dtype=np.float64)
    trimmed, peer = slice_dl_price_window(prices, pair, training_history_bars=288)
    assert len(trimmed) == 288
    assert len(peer) == 288
    assert float(trimmed[0]) == 112.0
    long_prices = np.arange(500, dtype=np.float64)
    short_peer = np.arange(400, dtype=np.float64)
    _, peer_tail = slice_dl_price_window(long_prices, short_peer, training_history_bars=288)
    assert len(peer_tail) == 288
    short = np.arange(50, dtype=np.float64)
    kept, no_peer = slice_dl_price_window(short, None, training_history_bars=288)
    assert len(kept) == 50
    assert no_peer is None


def test_parse_dl_params_and_optional_float():
    params = parse_dl_params(
        {
            "strong_signal_bypass": {"min_conviction_execute": 0.65, "min_edge_margin": 0.12},
        }
    )
    assert params["bypass_min_conviction"] == 0.65
    assert optional_float({}, "missing") is None
    full = parse_dl_params({"lookback": 32}, {"granularity": 300})
    assert full["training_history_bars"] == 288
    assert full["bars_per_day"] == 288
    bumped = parse_dl_params(
        {"lookback": 32, "deploy_gate": {"enabled": True, "mini_bars": 10}},
        {"granularity": 300},
    )
    assert bumped["deploy_gate"]["mini_bars"] == 37


def test_recovery_gating_active_when_pending_loss():
    orch = MagicMock()
    orch.risk_manager.pending_loss = {"R_50": 100.92}
    assert recovery_gating_active(orch) is True
    orch.risk_manager.pending_loss = {}
    assert recovery_gating_active(orch) is False


def test_resolve_dl_model_path_legacy():
    path = resolve_dl_model_path({"model_path": "data/legacy_model.pth"}, "X")
    assert path.name == "legacy_model.pth"
    templated = resolve_dl_model_path({"model_path_template": "data/dl/{symbol}.pth"}, "R_50")
    assert templated.name == "R_50.pth"


def test_predict_symbol_decision_gates_on_trade_score():
    params = parse_dl_params(
        {
            "min_conviction_execute": 0.56,
            "min_edge_margin": 0.06,
            "min_val_accuracy": 0.48,
            "require_regime_alignment": False,
            "min_direction_margin": 0.02,
        }
    )
    runtime = {
        "val_accuracy": 0.55,
        "calibrator": None,
        "val_brier": 0.2,
        "val_ece": 0.1,
        "lookback": 15,
    }
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.58, 0.58, 0.72),
    ):
        orch = type("O", (), {"config": {"deep_learning": {}}})()
        entry = predict_symbol_decision(
            orch,
            "R_50",
            MarketDirectionClassifier(input_dim=INPUT_DIM),
            np.zeros(80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["trade_score"] == 0.58
    assert entry["metrics"]["conviction"] == 0.58
    assert entry["metrics"]["raw_prob"] == 0.72
