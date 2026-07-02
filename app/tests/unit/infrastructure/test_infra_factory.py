"""Testes de stores de modelo e infra factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.factories.infra_factory import (
    close_infra_services,
    create_infra_services,
    validate_infra_services,
)
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@pytest.mark.asyncio
async def test_create_infra_disabled():
    services = create_infra_services({"infra": {"enabled": False}})
    assert services.enabled is False
    await validate_infra_services(services, {})


@pytest.mark.asyncio
async def test_validate_infra_fail_fast():
    services = create_infra_services({"infra": {"enabled": True, "fail_fast": True}})
    services.state_store.ping = AsyncMock(return_value=False)
    services.market_writer.ping = AsyncMock(return_value=True)
    services.model_store.head = AsyncMock(return_value=True)
    with pytest.raises(ConnectionError):
        await validate_infra_services(services, {"infra": {"enabled": True, "fail_fast": True}})


@pytest.mark.asyncio
async def test_local_model_store_roundtrip(tmp_path):
    store = LocalModelStore(tmp_path)
    src = tmp_path / "model.pth"
    src.write_bytes(b"ckpt")
    await store.upload("RDBEAR", src, arch="tcn", metadata={"val_accuracy": 0.6})
    dest = tmp_path / "cache.pth"
    assert await store.download_latest("RDBEAR", arch="tcn", dest=dest) is True
    assert dest.read_bytes() == b"ckpt"
    assert await store.head() is True


@pytest.mark.asyncio
async def test_minio_model_store_upload_download(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="dl-models",
        access_key="a",
        secret_key="b",
        secure=False,
    )
    client = MagicMock()
    client.bucket_exists.return_value = True
    store._client = client
    src = tmp_path / "m.pth"
    src.write_bytes(b"x")
    await store.upload("RDBEAR", src, arch="tcn", metadata={"a": 1})
    assert client.fput_object.called
    dest = tmp_path / "out.pth"
    with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
        assert await store.download_latest("RDBEAR", arch="tcn", dest=dest) is True
    with patch("asyncio.to_thread", new=AsyncMock(return_value=True)):
        assert await store.head() is True


@pytest.mark.asyncio
async def test_minio_head_creates_missing_bucket():
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="dl-models",
        access_key="a",
        secret_key="b",
        secure=False,
    )
    client = MagicMock()
    client.bucket_exists.return_value = False
    store._client = client

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.head() is True
    client.make_bucket.assert_called_once_with("dl-models")


@pytest.mark.asyncio
async def test_minio_head_inner_exception_returns_false():
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()
    client.bucket_exists.side_effect = RuntimeError("minio down")
    store._client = client

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.head() is False


@pytest.mark.asyncio
async def test_close_infra_services():
    services = create_infra_services({"infra": {"enabled": False}})
    services.state_store.close = AsyncMock()
    services.market_writer.flush = AsyncMock()
    services.market_writer.close = AsyncMock()
    services.model_store.close = AsyncMock()
    await close_infra_services(services)
