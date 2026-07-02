from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from torch import nn

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_artifacts import bootstrap_and_validate_models
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@pytest.mark.asyncio
async def test_minio_download_torchscript_and_sanity(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="dl-models",
        access_key="a",
        secret_key="b",
        secure=False,
    )
    store._client = MagicMock()
    dest = tmp_path / "RDBEAR_ts.pt"
    with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
        assert await store.download_torchscript("RDBEAR", arch="tcn", dest=dest) is True
    with patch(
        "src.infrastructure.storage.minio_model_store.verify_torchscript_artifact_async",
        new=AsyncMock(),
    ) as verify:
        await store.sanity_check_torchscript(
            dest,
            lookback=48,
            feature_dim=FEATURE_DIM,
            symbol="RDBEAR",
            manifest={"feature_dim": FEATURE_DIM},
        )
        verify.assert_awaited_once()


@pytest.mark.asyncio
async def test_minio_upload_includes_torchscript(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()
    client.bucket_exists.return_value = True
    store._client = client
    pth = tmp_path / "RDBEAR.pth"
    pth.write_bytes(b"ckpt")
    ts = tmp_path / "RDBEAR_ts.pt"
    ts.write_bytes(b"ts")
    with patch("asyncio.to_thread", new=AsyncMock(side_effect=lambda fn: fn())):
        await store.upload("RDBEAR", pth, arch="tcn", metadata={})
    assert client.fput_object.call_count >= 2


@pytest.mark.asyncio
async def test_bootstrap_and_validate_models_skips_when_no_ts(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = False
    orch.model_store = object()
    ckpt = tmp_path / "RDBEAR.pth"
    ckpt.write_bytes(b"x")
    with patch(
        "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
        new=AsyncMock(return_value=ckpt),
    ):
        await bootstrap_and_validate_models(orch)


@pytest.mark.asyncio
async def test_bootstrap_and_validate_models_runs_sanity_when_ts_present(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    ckpt = tmp_path / "RDBEAR.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "RDBEAR_ts.pt"
    ts_path.write_bytes(b"ts")
    store = MagicMock()
    store.download_torchscript = AsyncMock(return_value=True)
    store.sanity_check_torchscript = AsyncMock()
    orch.model_store = store
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new=AsyncMock(return_value=ckpt),
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=ts_path,
        ),
    ):
        await bootstrap_and_validate_models(orch)
    store.sanity_check_torchscript.assert_awaited_once()
    call_kw = store.sanity_check_torchscript.await_args.kwargs
    assert call_kw.get("manifest") is None


@pytest.mark.asyncio
async def test_bootstrap_loads_manifest_before_sanity(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    ckpt = tmp_path / "RDBEAR.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "RDBEAR_ts.pt"
    ts_path.write_bytes(b"ts")
    store = MagicMock()
    store.download_torchscript = AsyncMock(return_value=True)
    store.load_manifest = AsyncMock(return_value={"feature_dim": FEATURE_DIM, "lookback": 48})
    store.sanity_check_torchscript = AsyncMock()
    orch.model_store = store
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new=AsyncMock(return_value=ckpt),
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=ts_path,
        ),
    ):
        await bootstrap_and_validate_models(orch)
    store.load_manifest.assert_awaited_once()
    store.sanity_check_torchscript.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_model_store_torchscript_download_and_sanity(tmp_path):
    class _TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(FEATURE_DIM, 1)

        def forward(self, x):
            return self.fc(x[:, -1, :])

    store = LocalModelStore(tmp_path)
    ts_dir = tmp_path / "RDBEAR" / "tcn"
    ts_dir.mkdir(parents=True)
    ts_src = ts_dir / "latest_ts.pt"
    model = _TinyNet()
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 48, FEATURE_DIM), strict=False)
    traced.save(str(ts_src))
    dest = tmp_path / "out_ts.pt"
    assert await store.download_torchscript("RDBEAR", arch="tcn", dest=dest) is True
    await store.sanity_check_torchscript(
        dest,
        lookback=48,
        feature_dim=FEATURE_DIM,
        symbol="RDBEAR",
        manifest={"feature_dim": FEATURE_DIM, "lookback": 48},
    )


@pytest.mark.asyncio
async def test_minio_download_torchscript_inner_success(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()

    def _download(bucket, key, dest):
        Path(dest).write_bytes(b"ts")

    client.fget_object.side_effect = _download
    store._client = client
    dest = tmp_path / "ok_ts.pt"

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.download_torchscript("RDBEAR", arch="tcn", dest=dest) is True


@pytest.mark.asyncio
async def test_minio_download_torchscript_raises(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()
    client.fget_object.side_effect = RuntimeError("fail")
    store._client = client

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.download_torchscript("RDBEAR", arch="tcn", dest=tmp_path / "x_ts.pt") is False
