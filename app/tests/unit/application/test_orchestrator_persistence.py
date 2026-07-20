from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_persistence import (
    persist_full_state_unlocked,
    save_full_state,
)


@pytest.mark.asyncio
async def test_save_full_state_acquires_atomic_lock(orch_ready):
    orch = orch_ready
    orch.state.get_state = AsyncMock(return_value={"balance": 1000.0})
    with patch(
        "src.application.services.orchestrator.orchestrator_persistence.persist_full_state_unlocked",
        new_callable=AsyncMock,
    ) as persist_mock:
        assert await save_full_state(orch) is True
    persist_mock.assert_awaited_once_with(orch)
    assert not orch.state_mgr._state_lock.locked()


@pytest.mark.asyncio
async def test_save_full_state_soft_timeout_does_not_raise(orch_ready):
    orch = orch_ready
    await orch.state_mgr._state_lock.acquire()
    try:
        with patch(
            "src.application.services.orchestrator.orchestrator_atomic_state._STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS",
            0.05,
        ):
            assert await save_full_state(orch, raise_on_timeout=False) is False
    finally:
        orch.state_mgr._state_lock.release()


@pytest.mark.asyncio
async def test_save_full_state_raise_on_timeout(orch_ready):
    orch = orch_ready
    await orch.state_mgr._state_lock.acquire()
    try:
        with (
            patch(
                "src.application.services.orchestrator.orchestrator_atomic_state._STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS",
                0.05,
            ),
            pytest.raises(RuntimeError, match="STATE_LOCK_TIMEOUT"),
        ):
            await save_full_state(orch, raise_on_timeout=True)
    finally:
        orch.state_mgr._state_lock.release()


@pytest.mark.asyncio
async def test_save_full_state_reraises_non_timeout_runtime_error(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_persistence.orchestrator_atomic_state_context",
            side_effect=RuntimeError("OTHER_LOCK_ERROR"),
        ),
        pytest.raises(RuntimeError, match="OTHER_LOCK_ERROR"),
    ):
        await save_full_state(orch, raise_on_timeout=False)


@pytest.mark.asyncio
async def test_persist_full_state_unlocked_mirrors_balance(orch_ready):
    orch = orch_ready
    orch.state.balance = 1500.0
    orch.state.get_state = AsyncMock(return_value={"balance": 1500.0})
    orch.get_data_state_signature = MagicMock(return_value="")
    with patch.object(orch.state_store, "save_snapshot", new_callable=AsyncMock):
        await persist_full_state_unlocked(orch)
    assert orch.state_mgr.read_cached_balance() == pytest.approx(1500.0)
