from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.recovery_hurst_store import (
    increment_recovery_skip_counter,
    load_recovery_skip_counter,
    prepare_recovery_skip_counter,
    reset_recovery_skip_counter,
)
from src.domain.risk.recovery_hurst_decay import REDIS_SKIP_COUNTER_KEY


@pytest.mark.asyncio
async def test_recovery_skip_counter_redis_roundtrip():
    store = AsyncMock()
    store.get_string = AsyncMock(return_value=None)
    store.set_string = AsyncMock()

    assert await load_recovery_skip_counter(store) == 0
    assert await increment_recovery_skip_counter(store) == 1
    store.set_string.assert_awaited_with(REDIS_SKIP_COUNTER_KEY, "1")

    store.get_string = AsyncMock(return_value="3")
    assert await load_recovery_skip_counter(store) == 3
    assert await increment_recovery_skip_counter(store) == 4

    await reset_recovery_skip_counter(store)
    store.set_string.assert_awaited_with(REDIS_SKIP_COUNTER_KEY, "0")


@pytest.mark.asyncio
async def test_recovery_skip_counter_invalid_string():
    store = AsyncMock()
    store.get_string = AsyncMock(return_value="bad")
    assert await load_recovery_skip_counter(store) == 0


@pytest.mark.asyncio
async def test_prepare_recovery_skip_counter_loads_when_active():
    orch = MagicMock()
    store = AsyncMock()
    store.get_string = AsyncMock(return_value="4")
    orch.state_store = store
    count = await prepare_recovery_skip_counter(orch, recovery_active=True)
    assert count == 4
    assert orch._recovery_skip_counter == 4


@pytest.mark.asyncio
async def test_prepare_recovery_skip_counter_resets_when_inactive():
    orch = MagicMock()
    store = AsyncMock()
    store.set_string = AsyncMock()
    orch.state_store = store
    count = await prepare_recovery_skip_counter(orch, recovery_active=False)
    assert count == 0
    assert orch._recovery_skip_counter == 0


@pytest.mark.asyncio
async def test_recovery_skip_counter_without_store():
    assert await load_recovery_skip_counter(None) == 0
    assert await increment_recovery_skip_counter(None) == 1
    await reset_recovery_skip_counter(None)


@pytest.mark.asyncio
async def test_recovery_skip_counter_uses_incr_string_when_available():
    class _StoreWithIncr:
        def __init__(self):
            self.calls: list[str] = []

        async def incr_string(self, key: str) -> int:
            self.calls.append(key)
            return 7

    store = _StoreWithIncr()
    assert await increment_recovery_skip_counter(store) == 7
    assert store.calls == [REDIS_SKIP_COUNTER_KEY]
