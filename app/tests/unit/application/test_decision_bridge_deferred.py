from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_retrain import mark_force_retrain


@pytest.mark.asyncio
async def test_collect_skips_new_candle_train_on_fast_cycle(orch_ready_train):
    orch = orch_ready_train
    orch._dl_fast_cycle = True
    n = 1500
    ohlc = (np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n))
    runtime = {
        "model": MagicMock(),
        "norm_stats": object(),
        "last_candle_epoch": 99,
        "deploy_ok": True,
    }
    entry = {"direction": None, "metrics": {"gate_reason": "block", "execute": False}}

    with (
        patch("src.application.services.deep_learning.decision_bridge.load_symbol_close_ohlc", return_value=ohlc),
        patch("src.application.services.deep_learning.decision_bridge.get_symbol_runtime", return_value=runtime),
        patch("src.application.services.deep_learning.decision_bridge.candle_epoch", return_value=100),
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "new_candle"),
        ),
        patch("src.application.services.deep_learning.decision_bridge.predict_symbol_decision", return_value=entry),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)

    mock_enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_collect_defers_bootstrap_without_fast_cycle(orch_ready_train):
    orch = orch_ready_train
    n = 1500
    ohlc = (np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n))
    runtime = {
        "model": MagicMock(),
        "norm_stats": object(),
        "last_candle_epoch": 0,
        "deploy_ok": False,
    }
    entry = {"direction": None, "metrics": {"gate_reason": "block", "execute": False}}

    with (
        patch("src.application.services.deep_learning.decision_bridge.load_symbol_close_ohlc", return_value=ohlc),
        patch("src.application.services.deep_learning.decision_bridge.get_symbol_runtime", return_value=runtime),
        patch("src.application.services.deep_learning.decision_bridge.candle_epoch", return_value=100),
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch("src.application.services.deep_learning.decision_bridge.predict_symbol_decision", return_value=entry),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)

    mock_enqueue.assert_called()


@pytest.mark.asyncio
async def test_collect_bootstrap_only_enqueues_first_pending_symbol(orch_ready_train):
    orch = orch_ready_train
    n = 3000
    ohlc = (np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n))
    runtime = {
        "model": MagicMock(),
        "norm_stats": object(),
        "last_candle_epoch": 0,
        "deploy_ok": False,
    }
    entry = {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}

    with (
        patch("src.application.services.deep_learning.decision_bridge.load_symbol_close_ohlc", return_value=ohlc),
        patch("src.application.services.deep_learning.decision_bridge.get_symbol_runtime", return_value=runtime),
        patch("src.application.services.deep_learning.decision_bridge.candle_epoch", return_value=100),
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch("src.application.services.deep_learning.decision_bridge.predict_symbol_decision", return_value=entry),
        patch(
            "src.application.services.deep_learning.decision_bridge.training_priority_symbols",
            return_value=frozenset(orch.symbols),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)

    assert mock_enqueue.call_count == 1
    assert mock_enqueue.call_args.args[1] == orch.symbols[0]


@pytest.mark.asyncio
async def test_collect_skips_bootstrap_defer_after_initial_bootstrap(orch_ready_train):
    orch = orch_ready_train
    orch._dl_bootstrap_completed = True
    n = 3000
    ohlc = (np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n))
    runtime = {
        "model": MagicMock(),
        "norm_stats": object(),
        "last_candle_epoch": 0,
        "deploy_ok": False,
    }
    entry = {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}

    with (
        patch("src.application.services.deep_learning.decision_bridge.load_symbol_close_ohlc", return_value=ohlc),
        patch("src.application.services.deep_learning.decision_bridge.get_symbol_runtime", return_value=runtime),
        patch("src.application.services.deep_learning.decision_bridge.candle_epoch", return_value=100),
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch("src.application.services.deep_learning.decision_bridge.predict_symbol_decision", return_value=entry),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)

    mock_enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_collect_defers_retrain_when_fast_cycle(orch_ready_train):
    orch = orch_ready_train
    orch._dl_fast_cycle = True
    symbol = orch.symbols[0]
    mark_force_retrain(orch, symbol)
    n = 1500
    ohlc = (np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n), np.linspace(1.0, 2.0, n))
    runtime = {
        "model": MagicMock(),
        "norm_stats": object(),
        "last_candle_epoch": 99,
        "deploy_ok": True,
    }
    entry = {"direction": None, "metrics": {"gate_reason": "block", "execute": False}}

    with (
        patch("src.application.services.deep_learning.decision_bridge.load_symbol_close_ohlc", return_value=ohlc),
        patch("src.application.services.deep_learning.decision_bridge.get_symbol_runtime", return_value=runtime),
        patch("src.application.services.deep_learning.decision_bridge.candle_epoch", return_value=100),
        patch("src.application.services.deep_learning.decision_bridge.predict_symbol_decision", return_value=entry),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        await collect_deep_learning_decisions(orch)

    assert mock_enqueue.call_count >= 1
