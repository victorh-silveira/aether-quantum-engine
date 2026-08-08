"""Testes do RedisStateStore com cliente mockado."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.state.redis_state_store import RedisStateStore


def _pipeline_mock():
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.execute = AsyncMock()
    return pipe


@pytest.mark.asyncio
async def test_redis_save_and_load_snapshot():
    client = AsyncMock()
    client.ping.return_value = True
    client.get.return_value = json.dumps({"risk": {"consecutive_losses_linear": 2}})
    pipe = _pipeline_mock()
    client.pipeline = MagicMock(return_value=pipe)
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_snapshot({"risk": {"consecutive_losses_linear": 2}, "x": 1})
        loaded = await store.load_snapshot()
    assert loaded["risk"]["consecutive_losses_linear"] == 2


@pytest.mark.asyncio
async def test_redis_delete_string():
    client = AsyncMock()
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.delete_string("session:current:target_win")
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_hash_and_string_keys():
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.hgetall.return_value = {"a": "1"}
    client.get.return_value = "epoch"
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.set_hash("session:daily", {"a": 1})
        await store.set_string("bar_sig:R_10", "123")
        await store.set_string("cooldown:R_10", "1", ttl_seconds=30)
        assert await store.get_hash("session:daily") == {"a": "1"}
        assert await store.get_string("bar_sig:R_10") == "epoch"
        assert await store.ping() is True


@pytest.mark.asyncio
async def test_redis_debounce_flushes_pending():
    client = AsyncMock()
    pipe = _pipeline_mock()
    client.pipeline = MagicMock(return_value=pipe)
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=10.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_snapshot({"a": 1})
        await store.save_snapshot({"a": 2})
        await store.flush_snapshot()
    pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_redis_save_state_bundle_uses_pipeline():
    client = AsyncMock()
    pipe = _pipeline_mock()
    client.pipeline = MagicMock(return_value=pipe)
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_state_bundle(
            snapshot={"risk": {"consecutive_losses_linear": 1, "pending_loss": {"R_10": 2.5}}},
            session={"day_key": 1},
            market_sig="sig",
        )
    pipe.set.assert_called()
    pipe.hset.assert_called()
    pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_redis_pipeline_writes_pending_loss():
    client = AsyncMock()
    pipe = _pipeline_mock()
    client.pipeline = MagicMock(return_value=pipe)
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_snapshot({"risk": {"consecutive_losses_linear": 2, "pending_loss": {"R_10": 3.0}}})
    assert pipe.hset.call_args_list


@pytest.mark.asyncio
async def test_redis_from_url_uses_socket_timeouts():
    store = RedisStateStore(
        url="redis://127.0.0.1:6379/0",
        socket_connect_timeout=1.5,
        socket_timeout=3.5,
    )
    with patch("redis.asyncio.from_url", return_value=AsyncMock()) as from_url:
        await store.ping()
    assert from_url.call_args.kwargs["socket_connect_timeout"] == pytest.approx(1.5)
    assert from_url.call_args.kwargs["socket_timeout"] == pytest.approx(3.5)
