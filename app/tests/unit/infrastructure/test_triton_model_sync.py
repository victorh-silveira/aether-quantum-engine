from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import infer_symbols_async
from src.infrastructure.inference.triton_model_sync import (
    _files_identical,
    sync_all_symbols_to_triton,
    sync_symbol_torchscript_to_triton,
)
from tests.unit.infrastructure.test_torchscript_sanity import _trace_model


@pytest.mark.asyncio
async def test_infer_symbols_async_batch(tmp_path):
    tensors = {"RDBEAR": np.zeros((1, 4, 34), dtype=np.float32)}
    with patch(
        "src.infrastructure.inference.triton_inference_client.get_triton_grpc_client",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value.infer_symbols_concurrent = AsyncMock(return_value={"RDBEAR": 0.61})
        out = await infer_symbols_async({"infra": {"triton": {"enabled": True}}}, tensors)
    assert out["RDBEAR"] == pytest.approx(0.61)


@pytest.mark.asyncio
async def test_sync_symbol_torchscript_to_triton_writes_model(tmp_path):
    ts_path = tmp_path / "RDBEAR_ts.pt"
    _trace_model(ts_path, lookback=48)
    repo = tmp_path / "models"

    class _Store:
        pass

    ok, changed = await sync_symbol_torchscript_to_triton(
        _Store(),
        "RDBEAR",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    assert ok is True
    assert changed is True
    assert (repo / "RDBEAR" / "1" / "model.pt").is_file()
    assert (repo / "RDBEAR" / "config.pbtxt").is_file()


@pytest.mark.asyncio
async def test_sync_symbol_download_torchscript_fails(tmp_path):
    missing = tmp_path / "missing.pt"

    class _Store:
        async def download_torchscript(self, symbol, *, arch, dest):
            _ = (symbol, arch, dest)
            return False

    ok, changed = await sync_symbol_torchscript_to_triton(
        _Store(),
        "RDBEAR",
        arch="tcn",
        local_ts_path=missing,
        lookback=48,
        repo_path_override=tmp_path / "repo",
    )
    assert ok is False
    assert changed is False


@pytest.mark.asyncio
async def test_sync_symbol_skips_unchanged_torchscript(tmp_path):
    ts_path = tmp_path / "RDBEAR_ts.pt"
    _trace_model(ts_path, lookback=48)
    repo = tmp_path / "models"

    class _Store:
        pass

    ok1, changed1 = await sync_symbol_torchscript_to_triton(
        _Store(),
        "RDBEAR",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    ok2, changed2 = await sync_symbol_torchscript_to_triton(
        _Store(),
        "RDBEAR",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    assert ok1 is True and changed1 is True
    assert ok2 is True and changed2 is False


@pytest.mark.asyncio
async def test_sync_all_symbols_no_synced_models(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.model_store = MagicMock()
    orch.config = {
        "deep_learning": {"arch": "tcn"},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    with patch(
        "src.infrastructure.inference.triton_model_sync.sync_symbol_torchscript_to_triton",
        new_callable=AsyncMock,
        return_value=(False, False),
    ) as mock_one:
        await sync_all_symbols_to_triton(orch, repo_path_override=tmp_path)
    mock_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_all_symbols_without_triton_reload(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.model_store = MagicMock()
    orch.config = {
        "deep_learning": {"arch": "tcn"},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": False}},
    }
    with (
        patch(
            "src.infrastructure.inference.triton_model_sync.sync_symbol_torchscript_to_triton",
            new_callable=AsyncMock,
            return_value=(True, True),
        ),
        patch(
            "src.infrastructure.inference.triton_model_sync.wait_triton_models_stable",
            new_callable=AsyncMock,
        ) as mock_reload,
    ):
        await sync_all_symbols_to_triton(orch, repo_path_override=tmp_path)
    mock_reload.assert_not_awaited()


def test_files_identical_requires_matching_bytes(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    shorter = tmp_path / "short.pt"
    left.write_bytes(b"abc")
    right.write_bytes(b"abd")
    shorter.write_bytes(b"ab")
    assert _files_identical(left, right) is False
    assert _files_identical(tmp_path / "missing.pt", right) is False
    assert _files_identical(left, shorter) is False


@pytest.mark.asyncio
async def test_sync_all_symbols_raises_when_triton_not_ready(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.model_store = MagicMock()
    orch.config = {
        "deep_learning": {"arch": "tcn"},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    with (
        patch(
            "src.infrastructure.inference.triton_model_sync.sync_symbol_torchscript_to_triton",
            new_callable=AsyncMock,
            return_value=(True, True),
        ),
        patch(
            "src.infrastructure.inference.triton_model_sync.wait_triton_models_stable",
            new_callable=AsyncMock,
            return_value=False,
        ),
        pytest.raises(ConnectionError, match="nao ficaram prontos"),
    ):
        await sync_all_symbols_to_triton(orch, repo_path_override=tmp_path)
