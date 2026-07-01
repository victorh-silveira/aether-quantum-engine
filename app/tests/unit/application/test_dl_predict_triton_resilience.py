from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.triton_grpc_client import TritonInferenceTimeout


@pytest.mark.asyncio
async def test_predict_raw_prob_async_triton_timeout_fallback():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}}
    model = MagicMock()
    runtime = {
        "lookback": 4,
        "model": model,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "calibrator": None,
    }
    prices = np.linspace(1.0, 2.0, 20)
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_triton.infer_symbol_async",
            new_callable=AsyncMock,
            side_effect=TritonInferenceTimeout("timeout"),
        ),
        patch(
            "src.application.services.deep_learning.dl_predict_triton.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.71, 0.71),
        ),
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "R_10",
            prices,
            runtime,
            {"lookback": 4, "confidence_call_threshold": 0.6, "confidence_put_threshold": 0.4},
            granularity=60,
            call_threshold=0.6,
            put_threshold=0.4,
        )
    assert direction == TradeDirection.CALL
    assert prob == pytest.approx(0.71)


@pytest.mark.asyncio
async def test_predict_raw_prob_async_triton_timeout_without_local_model():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}}
    runtime = {
        "lookback": 4,
        "model": None,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "calibrator": None,
    }
    with patch(
        "src.application.services.deep_learning.dl_predict_triton.infer_symbol_async",
        new_callable=AsyncMock,
        side_effect=TritonInferenceTimeout("timeout"),
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "R_10",
            np.linspace(1.0, 2.0, 20),
            runtime,
            {"lookback": 4},
            granularity=60,
        )
    assert direction is None
    assert prob == 0.5
    assert raw == 0.5
