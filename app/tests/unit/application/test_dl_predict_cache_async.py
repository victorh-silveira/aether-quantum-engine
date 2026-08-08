from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.infrastructure.inference.triton_tensor_builder import PartialInferenceHistoryError


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_reuses_triton_cache():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 7
    orch._last_epoch = 0
    cached_entry = {"direction": "CALL", "metrics": {"raw_prob": 0.62}}
    orch._dl_prediction_cache = {
        "R_10": {
            "entry": cached_entry,
            "tensor_fingerprint": b"fp",
            "boundary_epoch": 100,
            "cycle_id": 7,
        },
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
            return_value=True,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.m5_boundary_epoch",
            return_value=100,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_inference_tensor",
            return_value=np.ones((1, 4, 34), dtype=np.float32),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.inference_tensor_fingerprint",
            return_value=b"fp",
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.predict_raw_prob_async",
            new_callable=AsyncMock,
        ) as infer_mock,
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
    assert entry is cached_entry
    infer_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_returns_cache_on_exception():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 3
    orch._last_epoch = 0
    cached_entry = {"direction": "PUT", "metrics": {"raw_prob": 0.41}}
    orch._dl_prediction_cache = {
        "R_10": {
            "entry": cached_entry,
            "tensor_fingerprint": b"x",
            "boundary_epoch": 1,
            "cycle_id": 3,
        }
    }
    runtime = {"lookback": 4, "val_accuracy": 0.5, "calibrator": None}
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_prediction_context",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.m5_boundary_epoch",
            return_value=1,
        ),
    ):
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
    assert entry is cached_entry


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_partial_history_uses_cache():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    orch._active_cycle_id = 2
    orch._last_epoch = 0
    cached_entry = {"direction": "PUT", "metrics": {"raw_prob": 0.44}}
    orch._dl_prediction_cache = {
        "R_10": {
            "entry": cached_entry,
            "tensor_fingerprint": b"x",
            "boundary_epoch": 1,
            "cycle_id": 2,
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
            "src.application.services.deep_learning.dl_predict_async.build_inference_tensor",
            side_effect=PartialInferenceHistoryError("short"),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.m5_boundary_epoch",
            return_value=1,
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
    assert entry is cached_entry


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_triton_stores_cycle_id():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
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
            "src.application.services.deep_learning.dl_predict_async.at_signature_boundary",
            return_value=True,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.m5_boundary_epoch",
            return_value=200,
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.build_inference_tensor",
            return_value=np.ones((1, 4, 34), dtype=np.float32),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.inference_tensor_fingerprint",
            return_value=b"new-fp",
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_async.predict_raw_prob_async",
            new_callable=AsyncMock,
            return_value=("CALL", 0.55, 0.52),
        ),
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
            {"lookback": 4, "implied_vol_bars": 60},
            0.0,
            granularity=900,
        )
    assert entry is fresh
    slot = orch._dl_prediction_cache["R_10"]
    assert slot["cycle_id"] == 11
    assert slot["boundary_epoch"] == 200
    assert slot["tensor_fingerprint"] == b"new-fp"
