from unittest.mock import MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@pytest.mark.asyncio
async def test_local_model_store_load_manifest(tmp_path):
    store = LocalModelStore(tmp_path)
    ts_dir = tmp_path / "R_10" / "tcn"
    ts_dir.mkdir(parents=True)
    (ts_dir / "manifest.json").write_text(
        '{"feature_dim": 34, "lookback": 48}',
        encoding="utf-8",
    )
    manifest = await store.load_manifest("R_10", arch="tcn")
    assert manifest["feature_dim"] == FEATURE_DIM


@pytest.mark.asyncio
async def test_local_model_store_load_manifest_invalid_json(tmp_path):
    store = LocalModelStore(tmp_path)
    ts_dir = tmp_path / "R_10" / "tcn"
    ts_dir.mkdir(parents=True)
    (ts_dir / "manifest.json").write_text("not-json", encoding="utf-8")
    assert await store.load_manifest("R_10", arch="tcn") == {}


@pytest.mark.asyncio
async def test_local_model_store_load_manifest_non_dict_payload(tmp_path):
    store = LocalModelStore(tmp_path)
    ts_dir = tmp_path / "R_10" / "tcn"
    ts_dir.mkdir(parents=True)
    (ts_dir / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    assert await store.load_manifest("R_10", arch="tcn") == {}


@pytest.mark.asyncio
async def test_minio_load_manifest_success():
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    payload = b'{"feature_dim": 34, "lookback": 48}'

    class _Response:
        def read(self):
            return payload

        def close(self):
            return None

        def release_conn(self):
            return None

    client = MagicMock()
    client.get_object.return_value = _Response()
    store._client = client

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        manifest = await store.load_manifest("R_10", arch="tcn")
    assert manifest["feature_dim"] == FEATURE_DIM


@pytest.mark.asyncio
async def test_minio_load_manifest_failure():
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("missing")
    store._client = client

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.load_manifest("R_10", arch="tcn") == {}
