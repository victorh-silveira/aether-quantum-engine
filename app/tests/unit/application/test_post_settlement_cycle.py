from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)


def test_schedule_skips_when_not_running():
    orch = MagicMock()
    orch.running = False
    orch._post_settlement_task = None
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        schedule_trading_cycle_after_settlement(orch)
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_run_post_settlement_breath_and_cycle_invokes_trading():
    orch = MagicMock()
    orch.running = True
    orch.state.active_contracts = {}
    orch.is_trading = False
    orch.config = {"orchestrator": {"post_settlement_breath_seconds": 0}}
    orch._run_trading_cycle_if_ready = AsyncMock()
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.sleep", new_callable=AsyncMock):
        await run_post_settlement_breath_and_cycle(orch)
    orch._run_trading_cycle_if_ready.assert_awaited_once()
