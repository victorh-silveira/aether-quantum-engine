from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_backfill import (
    _profit_from_row,
    backfill_contract_from_profit_table,
    backfill_pending_contracts,
    fetch_open_contract,
    reconcile_single_contract,
    settlement_payload_from_profit_row,
    subscribe_open_contract,
)
from src.infrastructure.handlers.trade_handler import _contract_duration_seconds


def test_settlement_payload_from_profit_row():
    payload = settlement_payload_from_profit_row(99, {"contract_id": 99, "profit": 1.92, "status": "won"})
    poc = payload["proposal_open_contract"]
    assert poc["contract_id"] == 99
    assert poc["is_settled"] == 1
    assert poc["profit"] == 1.92


@pytest.mark.asyncio
async def test_reconcile_single_contract_marks_ws_offline_on_timeout():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_request_timeout_seconds": 1.0}}}
    orch.ws = MagicMock()
    orch.ws.is_running = True
    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(side_effect=TimeoutError("timeout")),
        ),
        pytest.raises(TimeoutError),
    ):
        await reconcile_single_contract(orch, 42)
    assert orch.ws.is_running is False


@pytest.mark.asyncio
async def test_backfill_contract_from_profit_table_transient_error():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 10}}}
    orch.ws = MagicMock()
    orch.ws.is_running = True
    orch.ws.send = AsyncMock(side_effect=ConnectionError("offline"))
    ok = await backfill_contract_from_profit_table(orch, 1)
    assert ok is False
    assert orch.ws.is_running is False


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


def test_profit_from_row_sell_minus_buy():
    assert _profit_from_row({"sell_price": 4.26, "buy_price": 2.34}) == pytest.approx(1.92)


def test_contract_duration_seconds_units():
    assert _contract_duration_seconds({"duration": 15, "duration_unit": "m"}) == 900
    assert _contract_duration_seconds({"duration": 30, "duration_unit": "s"}) == 30
    assert _contract_duration_seconds({"duration": 5, "duration_unit": "t"}) == 10
    assert _contract_duration_seconds({"duration": 1, "duration_unit": "d"}) == 86400
    assert _contract_duration_seconds({"duration": 2, "duration_unit": "h"}) == 120


@pytest.mark.asyncio
async def test_subscribe_and_fetch_open_contract():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"proposal_open_contract": {"contract_id": 1, "status": "open"}})
    await subscribe_open_contract(ws, 1, timeout=5.0)
    poc = await fetch_open_contract(ws, 1, timeout=5.0, subscribe=False)
    assert poc["contract_id"] == 1


@pytest.mark.asyncio
async def test_reconcile_single_contract_settled():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_request_timeout_seconds": 5}}}
    orch.ws = AsyncMock()
    orch.state.active_contracts = {9: MagicMock()}
    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(return_value={"contract_id": 9, "is_settled": 1, "status": "won", "profit": 2.0}),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.process_contract_settlement",
            AsyncMock(),
        ) as mock_proc,
    ):
        ok = await reconcile_single_contract(orch, 9)
    assert ok is True
    mock_proc.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_pending_contracts_batch():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 5}}}
    with patch(
        "src.application.services.orchestrator.settlement_backfill.backfill_contract_from_profit_table",
        AsyncMock(side_effect=[True, False]),
    ):
        n = await backfill_pending_contracts(orch, [1, 2])
    assert n == 1


def test_settlement_payload_profit_only():
    payload = settlement_payload_from_profit_row(5, {"profit": -1.0})
    assert payload["proposal_open_contract"]["status"] == "lost"


def test_profit_from_row_zero_when_empty():
    assert _profit_from_row({}) == 0.0


@pytest.mark.asyncio
async def test_fetch_open_contract_error_and_invalid_payload():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"error": {"message": "x"}})
    assert await fetch_open_contract(ws, 1, timeout=1.0, subscribe=True) is None
    ws.send = AsyncMock(return_value={"proposal_open_contract": "bad"})
    assert await fetch_open_contract(ws, 1, timeout=1.0, subscribe=False) is None


@pytest.mark.asyncio
async def test_reconcile_falls_back_when_fetch_returns_none():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_request_timeout_seconds": 5}}}
    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(return_value=None),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.backfill_contract_from_profit_table",
            AsyncMock(return_value=False),
        ) as mock_bf,
    ):
        ok = await reconcile_single_contract(orch, 4)
    assert ok is False
    mock_bf.assert_awaited_once_with(orch, 4)


@pytest.mark.asyncio
async def test_reconcile_open_not_settled():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_request_timeout_seconds": 5}}}
    orch.ws = AsyncMock()
    with patch(
        "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
        AsyncMock(return_value={"contract_id": 1, "status": "open"}),
    ):
        ok = await reconcile_single_contract(orch, 1)
    assert ok is False


@pytest.mark.asyncio
async def test_backfill_finds_row_and_processes():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 5}}}
    orch.ws = AsyncMock()
    orch.ws.send = AsyncMock(
        return_value={"profit_table": {"transactions": [{"contract_id": 7, "profit": 2.0, "status": "won"}]}}
    )
    with patch(
        "src.application.services.orchestrator.settlement_backfill.process_contract_settlement",
        AsyncMock(),
    ) as mock_proc:
        ok = await backfill_contract_from_profit_table(orch, 7)
    assert ok is True
    mock_proc.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_profit_table_exception_and_bad_rows():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"settlement_profit_table_limit": 5}}}
    orch.ws = AsyncMock()
    orch.ws.send = AsyncMock(side_effect=RuntimeError("net"))
    assert await backfill_contract_from_profit_table(orch, 1) is False
    orch.ws.send = AsyncMock(return_value={"profit_table": {"transactions": ["x", {"contract_id": 2}]}})
    assert await backfill_contract_from_profit_table(orch, 1) is False
    orch.ws.send = AsyncMock(return_value={"profit_table": {}})
    assert await backfill_contract_from_profit_table(orch, 1) is False
