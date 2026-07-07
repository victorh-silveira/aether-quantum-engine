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
        await save_full_state(orch)
    persist_mock.assert_awaited_once_with(orch)
    assert not orch.state_mgr._state_lock.locked()


@pytest.mark.asyncio
async def test_persist_full_state_unlocked_mirrors_balance(orch_ready):
    orch = orch_ready
    orch.state.balance = 1500.0
    orch.state.get_state = AsyncMock(return_value={"balance": 1500.0})
    orch.get_data_state_signature = MagicMock(return_value="")
    with patch.object(orch.state_store, "save_snapshot", new_callable=AsyncMock):
        await persist_full_state_unlocked(orch)
    assert orch.state_mgr.read_cached_balance() == pytest.approx(1500.0)
