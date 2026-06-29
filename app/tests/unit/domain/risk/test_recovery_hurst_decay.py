from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.risk.recovery_hurst_decay import (
    REDIS_SKIP_COUNTER_KEY,
    effective_recovery_hurst_min,
    increment_recovery_skip_counter,
    load_recovery_skip_counter,
    prepare_recovery_skip_counter,
    reset_recovery_skip_counter,
    resolve_effective_hurst_min,
)


def test_effective_recovery_hurst_min_decay_to_floor():
    assert effective_recovery_hurst_min(0.58, 0) == 0.58
    assert effective_recovery_hurst_min(0.58, 3) == pytest.approx(0.55)
    assert effective_recovery_hurst_min(0.58, 8) == 0.50
    assert effective_recovery_hurst_min(0.58, 20) == 0.50


def test_resolve_effective_hurst_min_disabled():
    cfg = {
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_decay_enabled": False,
    }
    assert resolve_effective_hurst_min(cfg, 10) == 0.58


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
