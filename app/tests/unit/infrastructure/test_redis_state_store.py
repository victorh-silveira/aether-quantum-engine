"""Testes do RedisStateStore com cliente mockado."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.state.redis_state_store import RedisStateStore


@pytest.mark.asyncio
async def test_redis_save_and_load_snapshot():
    client = AsyncMock()
    client.ping.return_value = True
    client.get.return_value = json.dumps({"risk": {"consecutive_losses": 2}})
    store = RedisStateStore(url="redis://localhost:6379/0")
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_snapshot({"risk": {"consecutive_losses": 2}, "x": 1})
        loaded = await store.load_snapshot()
    assert loaded["risk"]["consecutive_losses"] == 2


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
    store = RedisStateStore(url="redis://localhost:6379/0", debounce_seconds=10.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.save_snapshot({"a": 1})
        await store.save_snapshot({"a": 2})
        await store.flush_snapshot()
    client.set.assert_called()
