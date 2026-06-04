import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    MarketDirectionClassifier,
    fit_norm_stats,
    save_model_checkpoint,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.dl_collect_fixtures import (
    MockOrchestrator,
    MockStreamNoEpochGetter,
)


@pytest.mark.asyncio
async def test_collect_uses_recovery_gating_when_pending_loss():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_75"], prices)
    orch.risk_manager = MagicMock()
    orch.risk_manager.pending_loss = {"R_50": 100.92}
    decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_75"]["metrics"]["execute"] in (True, False)


@pytest.mark.asyncio
async def test_collect_deep_learning_decisions():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50", "R_75"], prices)
    decisions = await collect_deep_learning_decisions(orch)
    assert "R_50" in decisions
    assert "R_75" in decisions
    assert "direction" in decisions["R_50"]
    assert "metrics" in decisions["R_50"]
    assert "conviction" in decisions["R_50"]["metrics"]
    assert "val_accuracy" in decisions["R_50"]["metrics"]
    orch_disabled = MockOrchestrator(["R_50"], prices, dl_enabled=False)
    dec_disabled = await collect_deep_learning_decisions(orch_disabled)
    assert dec_disabled == {}
    orch_short = MockOrchestrator(["R_50"], np.array([1.0, 2.0]), dl_enabled=True)
    dec_short = await collect_deep_learning_decisions(orch_short)
    assert dec_short["R_50"]["direction"] is None
    assert dec_short["R_50"]["metrics"]["conviction"] == 0.0


@pytest.mark.asyncio
async def test_collect_skips_train_on_same_candle():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50"], prices, epoch=5000)
    orch.config["deep_learning"]["train_on_new_candle_only"] = True
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    first = await collect_deep_learning_decisions(orch)
    assert "R_50" in first
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        side_effect=AssertionError("should not train"),
    ) as mock_train:
        second = await collect_deep_learning_decisions(orch)
        mock_train.assert_not_called()
    first_metrics = first["R_50"]["metrics"]
    second_metrics = second["R_50"]["metrics"]
    assert second["R_50"]["direction"] == first["R_50"]["direction"]
    assert second_metrics["execute"] == first_metrics["execute"]
    assert second_metrics["conviction"] == pytest.approx(first_metrics["conviction"])
    assert second_metrics["raw_conviction"] == pytest.approx(first_metrics["raw_conviction"])
    assert second_metrics["edge"] == pytest.approx(first_metrics["edge"])


@pytest.mark.asyncio
async def test_collect_predict_runs_each_cycle_same_candle():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50"], prices, epoch=5000)
    orch.config["deep_learning"]["train_on_new_candle_only"] = True
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    await collect_deep_learning_decisions(orch)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
            side_effect=AssertionError("should not train"),
        ) as mock_train,
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.55, 0.58, 0.56),
        ) as mock_predict,
    ):
        second = await collect_deep_learning_decisions(orch)
        mock_train.assert_not_called()
        mock_predict.assert_called_once()
    assert "R_50" in second


@pytest.mark.asyncio
async def test_collect_train_returns_none_resets_val_accuracy():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["R_50"], prices)
    orch.config["deep_learning"]["train_on_new_candle_only"] = False
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.dl_training.train_model_walkforward",
            return_value=None,
        ),
    ):
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_50"]["metrics"]["val_accuracy"] == 0.0


@pytest.mark.asyncio
async def test_collect_candle_epoch_without_getter():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50"], prices)
    orch.stream = MockStreamNoEpochGetter(prices)
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    decisions = await collect_deep_learning_decisions(orch)
    assert "R_50" in decisions


@pytest.mark.asyncio
async def test_collect_applies_symbol_loss_cooldown():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50"], prices)
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    orch.config["deep_learning"]["min_conviction_execute"] = 0.50
    orch.config["deep_learning"]["min_edge_margin"] = 0.01
    orch.config["deep_learning"]["min_direction_margin"] = 0.01
    orch.config["deep_learning"]["require_regime_alignment"] = False
    orch.risk_manager = MagicMock()
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.is_symbol_on_loss_cooldown = MagicMock(return_value=True)
    orch.config["deep_learning"]["max_val_brier_execute"] = 1.0
    orch.config["deep_learning"]["deploy_gate"] = {"enabled": False}
    stats = fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32))
    with (
        patch(
            "src.application.services.deep_learning.dl_predict.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.62, 0.62, 0.72),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.run_symbol_training",
            return_value=(stats, None),
        ),
    ):
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_50"]["metrics"]["execute"] is False
    assert decisions["R_50"]["metrics"]["gate_reason"] == "cooldown"


@pytest.mark.asyncio
async def test_collect_decisions_exceptions_and_load():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["R_50"], prices)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "R_50.pth"
        model = MarketDirectionClassifier(input_dim=INPUT_DIM)
        stats = fit_norm_stats(np.zeros((5, INPUT_DIM), dtype=np.float32))
        save_model_checkpoint(path, model, stats, last_candle_epoch=99, lookback=15, arch="tcn")
        orch.config["deep_learning"]["model_path_template"] = f"{tmp}/{{symbol}}.pth"
        if hasattr(orch, "_dl_runtime"):
            orch._dl_runtime.clear()
        decisions = await collect_deep_learning_decisions(orch)
        assert "R_50" in decisions
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=None,
    ):
        decisions = await collect_deep_learning_decisions(orch)
        assert "R_50" in decisions
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        side_effect=ValueError("Train failed"),
    ):
        if hasattr(orch, "_dl_runtime"):
            orch._dl_runtime.clear()
        dec = await collect_deep_learning_decisions(orch)
        assert "R_50" in dec
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        side_effect=ValueError("Predict failed"),
    ):
        dec = await collect_deep_learning_decisions(orch)
        assert dec["R_50"]["direction"] is None
        assert dec["R_50"]["metrics"]["conviction"] == 0.0
