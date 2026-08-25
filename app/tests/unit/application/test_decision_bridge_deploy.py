from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import (
    collect_deep_learning_decisions,
    runtime_in_training,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.dl_collect_fixtures import MockOrchestrator


@pytest.mark.asyncio
async def test_collect_blocks_execute_when_deploy_not_ok():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["R_10"], prices)
    orch.symbols = ["R_10"]
    orch.config["deep_learning"]["deploy_gate"] = {
        "enabled": True,
        "force_ok": False,
        "soft_min_val_accuracy": 0.99,
    }
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.52},
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(False, ""),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
        ) as mock_rt,
    ):
        mock_rt.return_value = {
            "model": MagicMock(),
            "norm_stats": MagicMock(),
            "val_accuracy": 0.52,
            "val_brier": 0.25,
            "calibrator": None,
            "lookback": 32,
            "deploy_ok": False,
            "deploy_win_rate": 0.0,
            "last_candle_epoch": 0,
            "session_trained": True,
        }
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_10"]["metrics"]["execute"] is False
    assert decisions["R_10"]["metrics"]["gate_reason"] == "deploy"


@pytest.mark.asyncio
async def test_collect_gives_training_slot_priority_to_untrained_symbols():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["R_10", "R_50"], prices, train_mode=True)
    orch.symbols = ["R_10", "R_50"]
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.55},
    }
    runtimes = {
        "R_10": {
            "model": MagicMock(),
            "norm_stats": MagicMock(),
            "val_accuracy": 0.55,
            "val_brier": 0.25,
            "calibrator": None,
            "lookback": 32,
            "deploy_ok": True,
            "deploy_win_rate": 0.6,
            "last_candle_epoch": 900,
            "session_trained": True,
        },
        "R_50": {
            "model": MagicMock(),
            "norm_stats": MagicMock(),
            "val_accuracy": 0.0,
            "val_brier": 1.0,
            "calibrator": None,
            "lookback": 32,
            "deploy_ok": False,
            "deploy_win_rate": 0.0,
            "last_candle_epoch": 0,
        },
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "new_candle"),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
            side_effect=lambda _orch, symbol, _cfg, _params: runtimes[symbol],
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.training_priority_symbols",
            return_value=frozenset({"R_10"}),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)
    enqueued = [call.args[1] for call in mock_enqueue.call_args_list]
    assert enqueued == ["R_10"]


@pytest.mark.asyncio
async def test_collect_enqueues_all_symbols_when_none_in_training():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["OTC_SPC"], prices, train_mode=True)
    orch.symbols = ["OTC_SPC"]
    orch.config["symbols"] = ["OTC_SPC"]
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.55},
    }
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.55,
        "val_brier": 0.25,
        "calibrator": None,
        "lookback": 32,
        "deploy_ok": True,
        "deploy_win_rate": 0.6,
        "last_candle_epoch": 900,
        "session_trained": True,
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "new_candle"),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
            return_value=runtime,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training",
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)
    enqueued = [call.args[1] for call in mock_enqueue.call_args_list]
    assert enqueued == ["OTC_SPC"]


def test_runtime_in_training_uses_brier_floor():
    params = {"brier_untrained_floor": 0.99}
    assert runtime_in_training({"val_brier": 1.0, "session_trained": True}, params) is True
    assert runtime_in_training({"val_brier": 0.99, "session_trained": True}, params) is True
    assert runtime_in_training({"val_brier": 0.25, "session_trained": True}, params) is False
    assert runtime_in_training({"session_trained": True}, params) is False


def test_runtime_in_training_requires_session_training():
    params = {"brier_untrained_floor": 0.99}
    assert runtime_in_training({"val_brier": 0.25}, params) is True
    assert runtime_in_training({"val_brier": 0.25, "session_trained": False}, params) is True


@pytest.mark.asyncio
async def test_collect_marks_symbol_in_training_until_first_valid_train():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["R_10"], prices)
    orch.symbols = ["R_10"]
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.0},
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(False, ""),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
        ) as mock_rt,
    ):
        mock_rt.return_value = {
            "model": MagicMock(),
            "norm_stats": MagicMock(),
            "val_accuracy": 0.0,
            "val_brier": 1.0,
            "calibrator": None,
            "lookback": 32,
            "deploy_ok": False,
            "deploy_win_rate": 0.0,
            "last_candle_epoch": 0,
        }
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_10"]["metrics"]["execute"] is False
    assert decisions["R_10"]["metrics"]["gate_reason"] == "training"
