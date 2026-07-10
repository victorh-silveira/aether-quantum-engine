from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.infrastructure.inference.triton_tensor_builder import PartialInferenceHistoryError


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_reuses_triton_cache():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    cached_entry = {"direction": "CALL", "metrics": {"raw_prob": 0.62}}
    orch._dl_prediction_cache = {
        "RDBULL": {"entry": cached_entry, "tensor_fingerprint": b"fp", "boundary_epoch": 100},
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
            "RDBULL",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            recovery_active=False,
            granularity=900,
        )
    assert entry is cached_entry
    infer_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_returns_cache_on_exception():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    cached_entry = {"direction": "PUT", "metrics": {"raw_prob": 0.41}}
    orch._dl_prediction_cache = {"RDBEAR": {"entry": cached_entry, "tensor_fingerprint": b"x", "boundary_epoch": 1}}
    runtime = {"lookback": 4, "val_accuracy": 0.5, "calibrator": None}
    with patch(
        "src.application.services.deep_learning.dl_predict_async.build_prediction_context",
        side_effect=RuntimeError("boom"),
    ):
        entry = await predict_symbol_decision_async(
            orch,
            "RDBEAR",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            recovery_active=False,
        )
    assert entry is cached_entry


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_partial_history_uses_cache():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    cached_entry = {"direction": "PUT", "metrics": {"raw_prob": 0.44}}
    orch._dl_prediction_cache = {"RDBEAR": {"entry": cached_entry, "tensor_fingerprint": b"x", "boundary_epoch": 1}}
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
            "RDBEAR",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            recovery_active=False,
        )
    assert entry is cached_entry


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_eager_cache_hit():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    cached_entry = {"direction": "CALL", "metrics": {"raw_prob": 0.7}}
    orch._dl_prediction_cache = {"RDBULL": {"entry": cached_entry, "tensor_fingerprint": b"x", "boundary_epoch": 1}}
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
            "src.application.services.deep_learning.dl_predict_async.eager_local_predict",
        ) as eager_mock,
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
            "RDBULL",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            recovery_active=False,
        )
    assert entry is cached_entry
    eager_mock.assert_not_called()


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
            "RDBEAR",
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
            "RDBEAR",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            runtime,
            {"lookback": 4},
            0.0,
            recovery_active=False,
        )
    assert entry["metrics"]["gate_reason"] == "predict_error"
