import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_settlement_queue import (
    _settlement_worker_loop,
)
from src.application.services.orchestrator.settlement_queue_ops import cancel_settlement_queue_fast


@pytest.mark.asyncio
async def test_settlement_worker_clears_pending_after_empty_queue(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._settlement_queue = asyncio.Queue()
    orch._settlement_pending_since = 99.0
    await orch._settlement_queue.put({"proposal_open_contract": {"contract_id": 1}})

    async def _stop_after_process(*_a, **_k):
        orch.running = False

    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue._maybe_run_orphan_cleaner",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.process_contract_settlement",
            new_callable=AsyncMock,
            side_effect=_stop_after_process,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue._known_contract_ids",
            return_value=[],
        ),
    ):
        await _settlement_worker_loop(orch)
    assert orch._settlement_pending_since == 0.0

    orch = orch_ready
    orch.running = True
    orch._settlement_queue = asyncio.Queue()
    orch._settlement_queue.put_nowait({"proposal_open_contract": {"contract_id": 1}})
    orch._settlement_queue.put_nowait({"proposal_open_contract": {"contract_id": 2}})

    async def _blocked_worker():
        await asyncio.sleep(3600)

    orch._settlement_worker_task = asyncio.create_task(_blocked_worker())
    cancel_settlement_queue_fast(orch)
    assert orch._settlement_queue.empty()
    assert orch._settlement_worker_task is None
