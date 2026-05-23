from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_backfill import (
    backfill_contract_from_profit_table,
    settlement_payload_from_profit_row,
)


def test_settlement_payload_from_profit_row():
    payload = settlement_payload_from_profit_row(99, {"contract_id": 99, "profit": 1.92, "status": "won"})
    poc = payload["proposal_open_contract"]
    assert poc["contract_id"] == 99
    assert poc["is_settled"] == 1
    assert poc["profit"] == 1.92


@pytest.mark.asyncio
async def test_backfill_contract_from_profit_table_error():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 10}}}
    orch.ws = AsyncMock()
    orch.ws.send = AsyncMock(return_value={"error": {"message": "x"}})
    ok = await backfill_contract_from_profit_table(orch, 1)
    assert ok is False


@pytest.mark.asyncio
async def test_backfill_contract_from_profit_table():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 10}}}
    orch.ws = AsyncMock()
    orch.ws.send = AsyncMock(
        return_value={
            "profit_table": {
                "transactions": [
                    {"contract_id": 76258194841, "profit": 1.92, "status": "won"},
                    {"contract_id": 1, "profit": -1.0, "status": "lost"},
                ]
            }
        }
    )
    with patch(
        "src.application.services.orchestrator.settlement_backfill.process_contract_settlement",
        new_callable=AsyncMock,
    ) as mock_proc:
        ok = await backfill_contract_from_profit_table(orch, 76258194841)
    assert ok is True
    mock_proc.assert_awaited_once()
