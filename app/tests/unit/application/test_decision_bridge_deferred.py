from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_retrain import mark_force_retrain


@pytest.mark.asyncio
async def test_collect_skips_new_candle_train_on_fast_cycle(orch_ready):
    orch = orch_ready
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
        patch(
            "src.application.services.deep_learning.decision_bridge.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread,
    ):
        await collect_deep_learning_decisions(orch)

    mock_enqueue.assert_not_called()
    mock_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_defers_retrain_when_fast_cycle(orch_ready):
    orch = orch_ready
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
        patch(
            "src.application.services.deep_learning.decision_bridge.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread,
    ):
        await collect_deep_learning_decisions(orch)

    assert mock_enqueue.call_count >= 1
    mock_thread.assert_not_awaited()
