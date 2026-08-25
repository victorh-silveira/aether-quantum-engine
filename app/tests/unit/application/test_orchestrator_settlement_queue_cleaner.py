from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_settlement_queue import (
    _maybe_run_orphan_cleaner,
)


@pytest.mark.asyncio
async def test_maybe_run_orphan_cleaner_after_tolerance_window(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = [7]
    orch._settlement_pending_since = 0.0
    with patch(
        "src.application.services.orchestrator.orchestrator_settlement_queue.time.monotonic",
        side_effect=[10.0, 700.0, 700.0],
    ):
        await _maybe_run_orphan_cleaner(orch)
        with patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=1,
        ) as cleaner:
            await _maybe_run_orphan_cleaner(orch)
        cleaner.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_run_orphan_cleaner_clears_pending_when_fully_reconciled(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = [7]
    orch._settlement_pending_since = 1.0
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.time.monotonic",
            return_value=700.0,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue._known_contract_ids",
            side_effect=[[7], [], []],
        ),
    ):
        await _maybe_run_orphan_cleaner(orch)
    assert orch._settlement_pending_since == 0.0
