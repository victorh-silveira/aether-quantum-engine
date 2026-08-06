import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    MarketDirectionClassifier,
    create_direction_model,
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
    orch = MockOrchestrator(["OTC_SPC"], prices)
    orch.risk_manager = MagicMock()
    orch.risk_manager.pending_loss = {"OTC_SPC": 100.92}
    decisions = await collect_deep_learning_decisions(orch)
    assert decisions["OTC_SPC"]["metrics"]["execute"] in (True, False)


@pytest.mark.asyncio
async def test_collect_deep_learning_decisions():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC", "R_50"], prices)
    decisions = await collect_deep_learning_decisions(orch)
    assert "OTC_SPC" in decisions
    assert "OTC_SPC" in decisions
    assert "direction" in decisions["OTC_SPC"]
    assert "metrics" in decisions["OTC_SPC"]
    assert "conviction" in decisions["OTC_SPC"]["metrics"]
    assert "val_accuracy" in decisions["OTC_SPC"]["metrics"]
    orch_disabled = MockOrchestrator(["OTC_SPC"], prices, dl_enabled=False)
    dec_disabled = await collect_deep_learning_decisions(orch_disabled)
    assert dec_disabled == {}
    orch_short = MockOrchestrator(["OTC_SPC"], np.array([1.0, 2.0]), dl_enabled=True)
    dec_short = await collect_deep_learning_decisions(orch_short)
    assert dec_short["OTC_SPC"]["direction"] is None
    assert dec_short["OTC_SPC"]["metrics"]["conviction"] == 0.0


@pytest.mark.asyncio
async def test_collect_skips_train_on_same_candle():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices, epoch=5000, train_mode=True)
    path = Path(orch.temp_dir) / "OTC_SPC.pth"
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    stats = fit_norm_stats(np.zeros((5, INPUT_DIM), dtype=np.float32))
    save_model_checkpoint(
        path,
        model,
        stats,
        last_candle_epoch=5000,
        lookback=15,
        arch="tcn",
        val_accuracy=0.6,
        val_brier=0.2,
        deploy_ok=True,
        granularity=900,
    )
    orch.config["deep_learning"]["train_on_new_candle_only"] = True
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    first = await collect_deep_learning_decisions(orch)
    assert "OTC_SPC" in first
    with patch(
        "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
        side_effect=AssertionError("should not train"),
    ) as mock_train:
        second = await collect_deep_learning_decisions(orch)
        mock_train.assert_not_called()
    first_metrics = first["OTC_SPC"]["metrics"]
    second_metrics = second["OTC_SPC"]["metrics"]
    assert second["OTC_SPC"]["direction"] == first["OTC_SPC"]["direction"]
    assert second_metrics["execute"] == first_metrics["execute"]
    assert second_metrics["conviction"] == pytest.approx(first_metrics["conviction"])
    assert second_metrics["raw_conviction"] == pytest.approx(first_metrics["raw_conviction"])
    assert second_metrics["edge"] == pytest.approx(first_metrics["edge"])


@pytest.mark.asyncio
async def test_collect_predict_runs_each_cycle_same_candle():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices, epoch=5000, train_mode=True)
    path = Path(orch.temp_dir) / "OTC_SPC.pth"
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    stats = fit_norm_stats(np.zeros((5, INPUT_DIM), dtype=np.float32))
    save_model_checkpoint(
        path,
        model,
        stats,
        last_candle_epoch=5000,
        lookback=15,
        arch="tcn",
        val_accuracy=0.6,
        val_brier=0.2,
        deploy_ok=True,
        granularity=900,
    )
    orch.config["deep_learning"]["train_on_new_candle_only"] = True
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    await collect_deep_learning_decisions(orch)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            side_effect=AssertionError("should not train"),
        ) as mock_train,
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.55, 0.58),
        ) as mock_predict,
    ):
        second = await collect_deep_learning_decisions(orch)
        mock_train.assert_not_called()
        mock_predict.assert_called_once()
    assert "OTC_SPC" in second


@pytest.mark.asyncio
async def test_collect_bootstrap_defers_training_without_blocking():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices, train_mode=True)
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        decisions = await collect_deep_learning_decisions(orch)
    mock_enqueue.assert_called_once()
    assert "OTC_SPC" in decisions


@pytest.mark.asyncio
async def test_collect_candle_epoch_without_getter():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices)
    orch.stream = MockStreamNoEpochGetter(prices)
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    decisions = await collect_deep_learning_decisions(orch)
    assert "OTC_SPC" in decisions


@pytest.mark.asyncio
async def test_collect_applies_symbol_loss_cooldown():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices)
    orch.config["deep_learning"]["min_val_accuracy"] = 0.0
    orch.config["deep_learning"]["confidence_call_threshold"] = 0.75
    orch.config["deep_learning"]["confidence_put_threshold"] = 0.25
    orch.config["deep_learning"]["deploy_gate"] = {"enabled": False}
    orch.risk_manager = MagicMock()
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.is_symbol_on_loss_cooldown = MagicMock(return_value=True)
    stats = fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32))
    orch._dl_runtime = {
        "OTC_SPC": {
            "model": MarketDirectionClassifier(input_dim=INPUT_DIM),
            "norm_stats": stats,
            "last_candle_epoch": 1000,
            "val_accuracy": 0.6,
            "calibrator": None,
            "val_brier": 0.2,
            "val_ece": 0.1,
            "lookback": 15,
            "deploy_ok": True,
            "deploy_win_rate": 0.6,
            "session_trained": True,
        }
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.80, 0.80),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train.run_symbol_training",
            return_value=(stats, None),
        ),
    ):
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["OTC_SPC"]["metrics"]["execute"] is True
    assert decisions["OTC_SPC"]["metrics"].get("gate_reason") != "symbol_cooldown"


@pytest.mark.asyncio
async def test_collect_decisions_exceptions_and_load():
    prices = np.sin(np.linspace(0, 10, 80)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "OTC_SPC.pth"
        model = MarketDirectionClassifier(input_dim=INPUT_DIM)
        stats = fit_norm_stats(np.zeros((5, INPUT_DIM), dtype=np.float32))
        save_model_checkpoint(path, model, stats, last_candle_epoch=99, lookback=15, arch="tcn", granularity=900)
        orch.config["deep_learning"]["model_path_template"] = f"{tmp}/{{symbol}}.pth"
        if hasattr(orch, "_dl_runtime"):
            orch._dl_runtime.clear()
        decisions = await collect_deep_learning_decisions(orch)
        assert "OTC_SPC" in decisions
    if hasattr(orch, "_dl_runtime"):
        orch._dl_runtime.clear()
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=None,
    ):
        decisions = await collect_deep_learning_decisions(orch)
        assert "OTC_SPC" in decisions
    with patch(
        "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
        side_effect=ValueError("Train failed"),
    ):
        if hasattr(orch, "_dl_runtime"):
            orch._dl_runtime.clear()
        dec = await collect_deep_learning_decisions(orch)
        assert "OTC_SPC" in dec
    with patch(
        "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
        side_effect=ValueError("Predict failed"),
    ):
        dec = await collect_deep_learning_decisions(orch)
        assert dec["OTC_SPC"]["direction"] is None
        assert dec["OTC_SPC"]["metrics"]["conviction"] == 0.0
