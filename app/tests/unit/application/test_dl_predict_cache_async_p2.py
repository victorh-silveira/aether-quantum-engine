from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.infrastructure.inference.triton_tensor_builder import PartialInferenceHistoryError


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_eager_skips_blind_cache():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 9
    orch._last_epoch = 0
    cached_entry = {"direction": "CALL", "metrics": {"raw_prob": 0.7}}
    orch._dl_prediction_cache = {
        "R_10": {
            "entry": cached_entry,
            "tensor_fingerprint": b"x",
            "boundary_epoch": 1,
            "cycle_id": 9,
        }
    }
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "val_accuracy": 0.5,
        "calibrator": None,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_async.at_signature_boundary",
            return_value=False,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.m5_boundary_epoch",
            return_value=1,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.eager_local_predict",
            return_value=("CALL", 0.61, 0.58),
        ) as eager_mock,
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_prediction_entry",
            return_value={"direction": "CALL", "fresh": True},
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
        )
    assert entry == {"direction": "CALL", "fresh": True}
    eager_mock.assert_called_once()


@pytest.mark.asyncio
async def test_predict_raw_prob_async_uses_prebuilt_tensor():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}}
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "calibrator": None,
    }
    tensor = np.ones((1, 4, 34), dtype=np.float32)
    with patch(
        "src.application.services.deep_learning.dl_predict_triton.infer_symbol_async",
        new_callable=AsyncMock,
        return_value=0.62,
    ) as infer_mock:
        await predict_raw_prob_async(
            orch,
            "R_10",
            np.linspace(1.0, 2.0, 20),
            runtime,
            {"lookback": 4},
            granularity=60,
            prebuilt_tensor=tensor,
        )
    infer_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_raises_without_cache_on_partial_history():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "val_accuracy": 0.5,
        "calibrator": None,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_inference_tensor",
            side_effect=PartialInferenceHistoryError("short"),
        ),
        patch("src.application.services.deep_learning.dl_predict_async.build_prediction_context") as ctx_mock,
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_decision_entry",
            return_value={"metrics": {"gate_reason": "predict_error"}},
        ),
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
        )
    assert entry["metrics"]["gate_reason"] == "predict_error"
