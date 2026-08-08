from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import infer_symbols_async
from src.infrastructure.inference.triton_model_sync import (
    _files_identical,
    _fsync_directory,
    _fsync_path,
    _repo_durability_barrier,
    sync_all_symbols_to_triton,
    sync_symbol_torchscript_to_triton,
)
from tests.unit.infrastructure.test_torchscript_sanity import _trace_model


@pytest.mark.asyncio
async def test_infer_symbols_async_batch(tmp_path):
    tensors = {"R_10": np.zeros((1, 4, 34), dtype=np.float32)}
    with patch(
        "src.infrastructure.inference.triton_inference_client.get_triton_grpc_client",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value.infer_symbols_concurrent = AsyncMock(return_value={"R_10": 0.61})
        out = await infer_symbols_async({"infra": {"triton": {"enabled": True}}}, tensors)
    assert out["R_10"] == pytest.approx(0.61)


@pytest.mark.asyncio
async def test_sync_symbol_torchscript_to_triton_writes_model(tmp_path):
    ts_path = tmp_path / "R_10_ts.pt"
    _trace_model(ts_path, lookback=48)
    repo = tmp_path / "models"

    class _Store:
        pass

    ok, changed = await sync_symbol_torchscript_to_triton(
        _Store(),
        "R_10",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    assert ok is True
    assert changed is True
    assert (repo / "R_10" / "1" / "model.pt").is_file()
    assert (repo / "R_10" / "config.pbtxt").is_file()


@pytest.mark.asyncio
async def test_sync_symbol_download_torchscript_fails(tmp_path):
    missing = tmp_path / "missing.pt"

    class _Store:
        async def download_torchscript(self, symbol, *, arch, dest):
            _ = (symbol, arch, dest)
            return False

    ok, changed = await sync_symbol_torchscript_to_triton(
        _Store(),
        "R_10",
        arch="tcn",
        local_ts_path=missing,
        lookback=48,
        repo_path_override=tmp_path / "repo",
    )
    assert ok is False
    assert changed is False


@pytest.mark.asyncio
async def test_sync_symbol_skips_unchanged_torchscript(tmp_path):
    ts_path = tmp_path / "R_10_ts.pt"
    _trace_model(ts_path, lookback=48)
    repo = tmp_path / "models"

    class _Store:
        pass

    ok1, changed1 = await sync_symbol_torchscript_to_triton(
        _Store(),
        "R_10",
        arch="tcn",
        local_ts_path=ts_path,
        lookback=48,
        repo_path_override=repo,
    )
    ok2, changed2 = await sync_symbol_torchscript_to_triton(
        _Store(),
        "R_10",
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
    orch.symbols = ["R_10"]
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
    orch.symbols = ["R_10"]
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


def test_fsync_helpers_tolerate_os_errors(tmp_path):
    missing = tmp_path / "missing.bin"
    _fsync_path(missing)
    with patch("src.infrastructure.inference.triton_model_sync.os.open", side_effect=OSError("denied")):
        _fsync_directory(tmp_path)
    with (
        patch("src.infrastructure.inference.triton_model_sync.os.open", return_value=7),
        patch("src.infrastructure.inference.triton_model_sync.os.fsync", side_effect=OSError("bad")),
        patch("src.infrastructure.inference.triton_model_sync.os.close") as close_mock,
    ):
        _fsync_directory(tmp_path)
        _fsync_path(tmp_path / "x")
    assert close_mock.call_count >= 2


def test_repo_durability_barrier_fsyncs_existing_artifacts(tmp_path):
    model_pt = tmp_path / "R_10" / "1" / "model.pt"
    pbtxt = tmp_path / "R_10" / "config.pbtxt"
    model_pt.parent.mkdir(parents=True)
    model_pt.write_bytes(b"pt")
    pbtxt.write_text("name: R_10\n", encoding="utf-8")
    with (
        patch("src.infrastructure.inference.triton_model_sync._fsync_path") as fsync_path,
        patch("src.infrastructure.inference.triton_model_sync._fsync_directory") as fsync_dir,
    ):
        _repo_durability_barrier(tmp_path, ["R_10", "MISSING"])
    assert fsync_path.call_count >= 2
    assert fsync_dir.call_count >= 2


@pytest.mark.asyncio
async def test_sync_all_symbols_raises_when_triton_not_ready(tmp_path):
    orch = MagicMock()
    orch.symbols = ["R_10"]
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
