"""Cobertura estendida de stores, Timescale e stream handler."""

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.redis_state_store import RedisStateStore
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@pytest.mark.asyncio
async def test_local_model_store_missing_download(tmp_path):
    store = LocalModelStore(tmp_path)
    dest = tmp_path / "missing.pth"
    assert await store.download_latest("X", arch="tcn", dest=dest) is False
    assert await store.download_torchscript("X", arch="tcn", dest=tmp_path / "missing_ts.pt") is False


@pytest.mark.asyncio
async def test_minio_upload_creates_bucket(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()
    client.bucket_exists.return_value = False
    store._client = client
    src = tmp_path / "a.pth"
    src.write_bytes(b"1")
    await store.upload("R_10", src, arch="tcn", metadata={"x": 1})
    client.make_bucket.assert_called_once()


@pytest.mark.asyncio
async def test_minio_download_fail_and_head_false():
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )

    with patch("asyncio.to_thread", new=AsyncMock(side_effect=[False, False])):
        assert await store.download_latest("R_10", arch="tcn", dest=Path("x.pth")) is False
        assert await store.head() is False


@pytest.mark.asyncio
async def test_timescale_writer_closed_and_worker_error():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=1.0)
    writer._closed = True
    await writer.enqueue_tick(symbol="R_10", epoch_ms=1, price=1.0)
    with patch.object(writer, "_flush_batches", AsyncMock(side_effect=RuntimeError("boom"))):
        writer._ensure_worker()
        await writer._run_worker()


@pytest.mark.asyncio
async def test_timescale_worker_batch_timeout():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=0.01, batch_limit=10)
    writer._closed = True
    writer._queue.put_nowait(("tick", (1, "R_10", 1, 1.0)))
    with patch.object(writer, "_flush_batches", AsyncMock()) as flush_mock:
        await writer._run_worker()
    flush_mock.assert_awaited()


@pytest.mark.asyncio
async def test_stream_handler_persist_bar_skips_without_writer():
    stream = StreamHandler(MagicMock(), ["R_10"], {}, market_writer=None)
    candle = MagicMock(open=1.0, high=2.0, low=0.5, close=1.5)
    await stream._persist_bar("R_10", 1, candle, None)


@pytest.mark.asyncio
async def test_timescale_enqueue_bar_when_closed():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    writer._closed = True
    await writer.enqueue_bar(symbol="R_10", bar={"epoch": 1})


@pytest.mark.asyncio
async def test_timescale_flush_with_active_worker():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=0.01)
    writer._worker = asyncio.create_task(asyncio.sleep(10))
    with patch.object(writer, "_flush_batches", AsyncMock()):
        await writer.flush()
    writer._worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await writer._worker


@pytest.mark.asyncio
async def test_redis_close_with_client():
    client = AsyncMock()
    store = RedisStateStore(url="redis://localhost/0")
    store._client = client
    await store.close()
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_json_state_store_non_dict_snapshot(tmp_path):
    path = tmp_path / "arr.json"
    path.write_text("[]", encoding="utf-8")
    store = JsonStateStore(path)
    assert await store.load_snapshot() is None


@pytest.mark.asyncio
async def test_json_empty_file_load_snapshot(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    store = JsonStateStore(path)
    assert await store.load_snapshot() is None
    assert await store.ping() is True


@pytest.mark.asyncio
async def test_redis_load_empty_and_non_dict():
    client = AsyncMock()
    client.get.return_value = None
    store = RedisStateStore(url="redis://localhost/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        assert await store.load_snapshot() is None
    client.get.return_value = "[]"
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        assert await store.load_snapshot() is None
    await store.close()


@pytest.mark.asyncio
async def test_redis_lazy_client_init():
    store = RedisStateStore(url="redis://localhost/0")
    with patch("redis.asyncio.from_url", return_value=AsyncMock()) as from_url:
        await store.ping()
    from_url.assert_called_once()


@pytest.mark.asyncio
async def test_stream_handler_tick_and_bar_with_writer():
    writer = AsyncMock()
    stream = StreamHandler(MagicMock(), ["R_10"], {}, market_writer=writer)
    await stream._on_tick({"tick": {"symbol": "R_10", "epoch": 1, "quote": 1.5}})
    candle = MagicMock(open=1.0, high=2.0, low=0.5, close=1.5)
    micro = MagicMock(tick_count=2, mean_inter_tick_ms=1.0, price_velocity=0.1)
    await stream._persist_bar("R_10", 10, candle, micro)
    writer.enqueue_tick.assert_awaited_once()
    writer.enqueue_bar.assert_awaited_once()


@pytest.mark.asyncio
async def test_timescale_full_lifecycle():
    conn = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    with patch("asyncpg.create_pool", AsyncMock(return_value=pool)):
        writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=0.02, batch_limit=1)
        await writer.enqueue_tick(symbol="R_10", epoch_ms=1000, price=1.0)
        await writer.enqueue_bar(symbol="R_10", bar={"epoch": 1, "granularity": 180, "open": 1.0})
        await asyncio.sleep(0.05)
        await writer.flush()
        assert await writer.ping() is True
        await writer.close()


@pytest.mark.asyncio
async def test_timescale_worker_cancelled():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    writer._worker = asyncio.create_task(writer._run_worker())
    writer._worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer._worker


@pytest.mark.asyncio
async def test_minio_download_inner_success(tmp_path):
    store = MinioModelStore(
        endpoint="localhost:9000",
        bucket="b",
        access_key="a",
        secret_key="s",
        secure=False,
    )
    client = MagicMock()

    def _download(bucket, key, dest):
        Path(dest).write_bytes(b"1")

    client.fget_object.side_effect = _download
    store._client = client
    dest = tmp_path / "ok.pth"

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.download_latest("R_10", arch="tcn", dest=dest) is True


@pytest.mark.asyncio
async def test_minio_head_inner_true():
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

    def _thread_run(fn):
        return fn()

    with patch("asyncio.to_thread", side_effect=_thread_run):
        assert await store.head() is True


@pytest.mark.asyncio
async def test_minio_download_raises(tmp_path):
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
        assert await store.download_latest("R_10", arch="tcn", dest=tmp_path / "x.pth") is False
    await store.close()


@pytest.mark.asyncio
async def test_timescale_worker_logs_errors():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=0.01, batch_limit=1)
    writer._queue.put_nowait(("tick", (1, "R_10", 1, 1.0)))
    with patch.object(writer, "_flush_batches", AsyncMock(side_effect=RuntimeError("db"))):
        await writer._run_worker()


@pytest.mark.asyncio
async def test_timescale_flush_bar_only():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    writer._queue.put_nowait(("bar", (1, "R_10", 1, 180, 1, 1, 1, 1, 1, 1, 1)))
    with patch.object(writer, "_flush_batches", AsyncMock()) as flush_mock:
        await writer.flush()
    flush_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_timescale_flush_tick_only():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    writer._queue.put_nowait(("tick", (1, "R_10", 1, 1.0)))
    with patch.object(writer, "_flush_batches", AsyncMock()) as flush_mock:
        await writer.flush()
    flush_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_timescale_worker_zero_timeout_break():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=50.0, batch_limit=5)
    writer._closed = True
    writer._queue.put_nowait(("tick", (1, "R_10", 1, 1.0)))
    loop = asyncio.get_running_loop()
    times = [100.0, 100.0, 100.1]

    async def fake_wait_for(coro, timeout):
        return await coro

    with (
        patch.object(loop, "time", side_effect=times),
        patch("asyncio.wait_for", side_effect=fake_wait_for),
        patch.object(writer, "_flush_batches", AsyncMock()) as flush_mock,
    ):
        await writer._run_worker()
    flush_mock.assert_awaited_once()
