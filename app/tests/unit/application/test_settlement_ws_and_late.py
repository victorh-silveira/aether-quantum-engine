from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_logic import process_late_settlement_from_payload
from src.application.services.orchestrator.settlement_ws_queries import fetch_portfolio, fetch_profit_table
from src.application.services.orchestrator.trading_cycle_entry import trading_cycle_entry_allowed


@pytest.mark.asyncio
async def test_fetch_portfolio_and_profit_table():
    ws = AsyncMock()
    ws.send = AsyncMock(
        side_effect=[
            {"portfolio": {"contracts": [{"contract_id": 1}]}},
            {"profit_table": {"transactions": [{"contract_id": 2, "profit": 1.0}]}},
        ]
    )
    portfolio = await fetch_portfolio(ws, timeout=1.0)
    rows = await fetch_profit_table(ws, limit=10, timeout=1.0)
    assert portfolio[0]["contract_id"] == 1
    assert rows[0]["contract_id"] == 2


@pytest.mark.asyncio
async def test_fetch_portfolio_error_returns_empty():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"error": {"message": "x"}})
    assert await fetch_portfolio(ws, timeout=1.0) == []


def test_trading_cycle_blocked_while_reconciliation_pending():
    orch = MagicMock()
    orch._reconciliation_pending = True
    assert trading_cycle_entry_allowed(orch) is False


@pytest.mark.asyncio
async def test_process_late_settlement_early_returns():
    orch = MagicMock()
    orch._save_full_state = AsyncMock()
    orch._persist_full_state_unlocked = AsyncMock()
    await process_late_settlement_from_payload(orch, {"is_settled": 0})
    await process_late_settlement_from_payload(orch, {"is_settled": 1})
    orch._save_full_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_portfolio_invalid_shapes():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"portfolio": "bad"})
    assert await fetch_portfolio(ws, timeout=1.0) == []

    ws.send = AsyncMock(return_value={"portfolio": {"contracts": "bad"}})
    assert await fetch_portfolio(ws, timeout=1.0) == []

    ws.send = AsyncMock(return_value={"portfolio": {"contracts": [{"contract_id": 1}, "x"]}})
    assert await fetch_portfolio(ws, timeout=1.0) == [{"contract_id": 1}]


@pytest.mark.asyncio
async def test_fetch_profit_table_error_returns_empty():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"error": {"message": "x"}})
    assert await fetch_profit_table(ws, limit=5, timeout=1.0) == []

    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"profit_table": []})
    assert await fetch_profit_table(ws, limit=5, timeout=1.0) == []


@pytest.mark.asyncio
async def test_process_late_settlement_from_payload():
    orch = MagicMock()
    orch._contract_cycle = {303: 7}
    orch._buffer_result_logs = False
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.active_contract_ids = []
    orch.state.active_contracts = {}
    orch.running = True
    orch._persist_full_state_unlocked = AsyncMock()
    orch.schedule_trading_cycle_after_settlement = MagicMock()
    poc = {
        "contract_id": 303,
        "is_settled": 1,
        "status": "won",
        "profit": 2.0,
        "underlying": "R_10",
    }
    with (
        patch(
            "src.application.services.orchestrator.settlement_logic._process_contract_outcome",
        ) as outcome,
        patch(
            "src.application.services.orchestrator.settlement_logic.reset_recovery_skip_counter_for_orch",
            new=AsyncMock(),
        ),
    ):
        await process_late_settlement_from_payload(orch, poc)
    outcome.assert_called_once()
    orch._persist_full_state_unlocked.assert_awaited_once()
