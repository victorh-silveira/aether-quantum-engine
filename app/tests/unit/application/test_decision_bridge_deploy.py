from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.domain.models.trade import TradeDirection
from tests.unit.application.dl_collect_fixtures import MockOrchestrator


@pytest.mark.asyncio
async def test_collect_blocks_execute_when_deploy_not_ok():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["RDBULL"], prices)
    orch.symbols = ["RDBULL"]
    orch.config["deep_learning"]["deploy_gate"] = {"enabled": True, "soft_min_val_accuracy": 0.99}
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
        }
        decisions = await collect_deep_learning_decisions(orch)
    assert decisions["RDBULL"]["metrics"]["execute"] is False
    assert decisions["RDBULL"]["metrics"]["gate_reason"] == "deploy"
