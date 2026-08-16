from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_error_returns_predict_error():
    orch = MagicMock()
    orch.config = {"infra": {}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 3
    orch._last_epoch = 0
    orch._dl_prediction_cache = {
        "R_10": {
            "entry": {"direction": "PUT", "metrics": {"raw_prob": 0.41}},
            "tensor_fingerprint": b"x",
            "boundary_epoch": 1,
            "cycle_id": 3,
        }
    }
    runtime = {"lookback": 4, "val_accuracy": 0.5, "calibrator": None}
    with patch(
        "src.application.services.deep_learning.dl_predict_async.build_prediction_context",
        side_effect=RuntimeError("boom"),
    ):
        entry = await predict_symbol_decision_async(
            orch,
            "R_10",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4, "contract_duration": 1500},
            0.0,
        )
    assert entry["metrics"]["gate_reason"] == "predict_error"
    assert entry["metrics"].get("execute") is False


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_eager_always_infers():
    orch = MagicMock()
    orch.config = {"infra": {}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 11
    orch._last_epoch = 0
    orch._dl_prediction_cache = {}
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "val_accuracy": 0.5,
        "calibrator": None,
    }
    fresh = {"direction": "CALL", "metrics": {"raw_prob": 0.55}}
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_async.eager_local_predict",
            return_value=("CALL", 0.55, 0.52),
        ) as eager_mock,
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_prediction_entry",
            return_value=fresh,
        ),
        patch("src.application.services.deep_learning.dl_predict_async.build_prediction_context") as ctx_mock,
    ):
        ctx_mock.return_value = {
            "gran": 900,
            "series": {},
            "dynamic": {},
            "dynamic_cfg": {},
            "call_threshold": 0.6,
            "put_threshold": 0.4,
            "exec_cfg": {},
        }
        entry = await predict_symbol_decision_async(
            orch,
            "R_10",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            granularity=900,
        )
    assert entry is fresh
    eager_mock.assert_called_once()
    assert "R_10" not in orch._dl_prediction_cache
