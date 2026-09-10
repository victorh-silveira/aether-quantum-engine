"""Cobertura residual para commit (SSOT stake 2%, majority, R_10)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import guard_inference_price_history
from src.application.services.deep_learning.dl_params import resolve_inference_history_bars
from src.application.services.deep_learning.dl_symbol_runtime import candle_epoch
from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.application.services.execution_gate_verdict import metrics_block_execution
from src.application.services.loss_classifier_vectors import store_loss_feature_vector
from src.application.services.orchestrator.settlement_outcome import _feed_loss_classifier_learn
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import finalize_stake_with_min


def test_guard_inference_partial_history_warnings(caplog):
    logger = logging.getLogger("cov.dl")
    prices = np.linspace(1.0, 2.0, 40)
    params = {"lookback": 30, "inference_history_bars": 80}
    with caplog.at_level(logging.WARNING):
        assert guard_inference_price_history(prices, prices, params, min_len=50, symbol="R_10", logger=logger) is None
    assert "Historico parcial" in caplog.text
    assert "Inferencia com historico parcial" in caplog.text


def test_resolve_inference_history_bars_without_windows():
    assert resolve_inference_history_bars({"lookback": 10, "implied_vol_bars": 20, "granularity": 900}) >= 288 + 10 + 16


def test_candle_epoch_micro_and_missing_stream():
    assert candle_epoch(SimpleNamespace(stream=None), "R_10") == 0
    stream = SimpleNamespace(get_last_micro_candle_epoch=lambda _s: None)
    assert candle_epoch(SimpleNamespace(stream=stream), "R_10", timeframe="micro") == 0
    stream2 = SimpleNamespace(get_last_micro_candle_epoch=lambda _s: 99)
    assert candle_epoch(SimpleNamespace(stream=stream2), "R_10", timeframe="micro") == 99


def test_finalize_execution_metrics_pending_total_errors():
    metrics = {
        "raw_prob": 0.62,
        "calibrated_prob": 0.61,
        "conviction": 0.62,
        "trade_score": 0.62,
    }
    entry = {"metrics": metrics}
    bad_total = MagicMock(side_effect=ValueError("x"))
    orch = SimpleNamespace(risk_manager=SimpleNamespace(pending_loss_total=bad_total, pending_loss=None), stream=None)
    with (
        patch("src.application.services.execution_direction_resolver.compute_scale_directions"),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.6),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=lambda *a, **k: None,
        ),
    ):
        _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="R_10",
            orch=orch,
        )
    assert metrics.get("pending_loss_total") == 0.0

    metrics2 = dict(metrics)
    entry2 = {"metrics": metrics2}
    orch2 = SimpleNamespace(risk_manager=SimpleNamespace(pending_loss={"a": "bad"}), stream=None)
    with (
        patch("src.application.services.execution_direction_resolver.compute_scale_directions"),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.6),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=lambda *a, **k: None,
        ),
    ):
        _finalize_execution_metrics(
            entry2,
            metrics2,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="R_10",
            orch=orch2,
        )
    assert metrics2.get("pending_loss_total") == 0.0


def test_finalize_stake_float_edge_returns_zero():
    assert finalize_stake_with_min(0.5, 1.0, 1.0 - 1e-13, 0.6, recovery_linear=False) == 0.0


def test_metrics_block_execution_non_dict():
    assert metrics_block_execution(None) is False
    assert metrics_block_execution("x") is False
    assert metrics_block_execution({"execution_candidate_ready": False}) is True
    assert metrics_block_execution({"signal_status": "SKIP:LOSS_CLF"}) is True


def test_store_loss_feature_vector_rejects_empty():
    orch = SimpleNamespace()
    store_loss_feature_vector(orch, "", [1.0])
    store_loss_feature_vector(orch, "R_10", [])
    assert getattr(orch, "_loss_clf_vectors", {}) == {}


def test_feed_loss_classifier_learn_skip_paths(caplog):
    orch = MagicMock()
    orch._loss_clf_vectors = {"R_10": [0.1] * 24, "cid:7": [0.2] * 24}
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
        return_value=None,
    ):
        _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=7)
    orch._loss_clf_vectors = {"R_10": [0.1] * 24, "cid:8": [0.2] * 24}
    with (
        patch(
            "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
            return_value={"skipped": True, "error": "buffer"},
        ),
        caplog.at_level(logging.WARNING),
    ):
        _feed_loss_classifier_learn(orch, "R_10", won=False, contract_id=8)
    assert "LEARN falhou" in caplog.text
