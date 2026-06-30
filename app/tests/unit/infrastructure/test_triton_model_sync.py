from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import infer_symbols_async
from src.infrastructure.inference.triton_model_sync import sync_symbol_torchscript_to_triton
from tests.unit.infrastructure.test_torchscript_sanity import _trace_model


@pytest.mark.asyncio
async def test_infer_symbols_async_batch(tmp_path):
    tensors = {"R_10": np.zeros((1, 4, 34), dtype=np.float32)}
    with patch(
        "src.infrastructure.inference.triton_inference_client.infer_symbol_async",
        new_callable=AsyncMock,
        return_value=0.61,
    ):
        out = await infer_symbols_async({"infra": {"triton": {"enabled": True}}}, tensors)
    assert out["R_10"] == pytest.approx(0.61)


@pytest.mark.asyncio
async def test_sync_symbol_torchscript_to_triton_writes_model(tmp_path):
    ts_path = tmp_path / "R_10_ts.pt"
    _trace_model(ts_path, lookback=48)
    repo = tmp_path / "models"

    class _Store:
        pass

    ok = await sync_symbol_torchscript_to_triton(
        _Store(),
        "R_10",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    assert ok is True
    assert (repo / "R_10" / "1" / "model.pt").is_file()
    assert (repo / "R_10" / "config.pbtxt").is_file()
