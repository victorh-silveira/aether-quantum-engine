"""Concorrencia entre stream, reconexao e liquidacao tardia."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_settlement_queue import enqueue_contract_settlement
from src.application.services.orchestrator.settlement_logic import (
    process_contract_settlement,
    process_late_settlement_from_payload,
)
from src.application.services.orchestrator.settlement_reconciliation import (
    ReconciliationResult,
    _settle_from_profit_burst,
)
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_stream_and_late_settlement_apply_same_contract_once():
    TradingState.reset()
    state = TradingState()
    contract = SimpleNamespace(contract_id=501)
    await state.add_contract(contract)
    started = asyncio.Event()
    release = asyncio.Event()

    async def process_once(*_args):
        started.set()
        await release.wait()

    orch = SimpleNamespace(
        ws=SimpleNamespace(is_running=True),
        state=state,
        risk_manager=SimpleNamespace(contract_to_symbol={501: "R_10"}, active_contract_ids=[501]),
    )
    poc = {"contract_id": 501, "status": "won", "is_settled": 1, "profit": 2.0}
    with patch(
        "src.application.services.orchestrator.settlement_logic._process_confirmed_settlement",
        new=AsyncMock(side_effect=process_once),
    ) as process:
        stream = asyncio.create_task(process_contract_settlement(orch, {"proposal_open_contract": poc}))
        await started.wait()
        late = asyncio.create_task(process_late_settlement_from_payload(orch, poc))
        release.set()
        await asyncio.gather(stream, late)
    process.assert_awaited_once()
    assert orch._settled_contract_ids == {501}
    assert not state.active_contracts


@pytest.mark.asyncio
async def test_duplicate_late_result_does_not_repeat_outcome():
    TradingState.reset()
    orch = MagicMock()
    orch.state = TradingState()
    orch.risk_manager.contract_to_symbol = {502: "R_25"}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.pending_loss = {}
    orch._contract_cycle = {}
    orch._persist_full_state_unlocked = AsyncMock()
    orch.running = False
    poc = {"contract_id": 502, "status": "lost", "is_settled": 1, "profit": -2.0}
    with patch("src.application.services.orchestrator.settlement_logic._process_contract_outcome") as outcome:
        await asyncio.gather(
            process_late_settlement_from_payload(orch, poc),
            process_late_settlement_from_payload(orch, poc),
        )
    outcome.assert_called_once()
    orch._persist_full_state_unlocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_late_callback_with_active_contract_uses_confirmed_path():
    TradingState.reset()
    state = TradingState()
    await state.add_contract(SimpleNamespace(contract_id=506))
    orch = SimpleNamespace(state=state)
    poc = {"contract_id": 506, "status": "won", "is_settled": 1, "profit": 2.0}
    with patch(
        "src.application.services.orchestrator.settlement_logic._process_confirmed_settlement",
        new=AsyncMock(),
    ) as confirmed:
        await process_late_settlement_from_payload(orch, poc)
        await process_late_settlement_from_payload(orch, poc)
    confirmed.assert_awaited_once()
    assert orch._settled_contract_ids == {506}


@pytest.mark.asyncio
async def test_stream_duplicate_and_missing_active_contract_use_shared_guard():
    TradingState.reset()
    state = TradingState()
    orch = SimpleNamespace(
        state=state,
        ws=SimpleNamespace(is_running=True),
        risk_manager=SimpleNamespace(contract_to_symbol={507: "R_10"}, active_contract_ids=[]),
    )
    poc = {"contract_id": 507, "status": "won", "is_settled": 1, "profit": 2.0}
    with patch(
        "src.application.services.orchestrator.settlement_logic._process_late_settlement_once",
        new=AsyncMock(),
    ) as late:
        await process_contract_settlement(orch, {"proposal_open_contract": poc})
        await process_contract_settlement(orch, {"proposal_open_contract": poc})
    late.assert_awaited_once()
    assert orch._settled_contract_ids == {507}


@pytest.mark.asyncio
async def test_untracked_late_result_is_ignored():
    TradingState.reset()
    orch = SimpleNamespace(
        state=TradingState(),
        risk_manager=SimpleNamespace(contract_to_symbol={}, active_contract_ids=[]),
    )
    await process_late_settlement_from_payload(
        orch, {"contract_id": 503, "status": "won", "is_settled": 1, "profit": 2.0}
    )
    assert orch._settled_contract_ids == set()


@pytest.mark.asyncio
async def test_profit_table_reconciliation_keeps_active_contract_until_settlement():
    orch = MagicMock()
    orch.state.active_contracts = {504: object()}
    orch.risk_manager.active_contract_ids = [504]
    orch.risk_manager.contract_to_symbol = {504: "R_10"}
    result = ReconciliationResult()
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_profit_table",
            new=AsyncMock(return_value=[{"contract_id": 504, "profit": 2.0}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.process_contract_settlement",
            new=AsyncMock(),
        ) as settle,
    ):
        await _settle_from_profit_burst(orch, open_ids=set(), settled_ids=set(), timeout=1.0, result=result)
    settle.assert_awaited_once()
    orch.state.finalize_contract.assert_not_called()
    assert result.settled_count == 1


@pytest.mark.asyncio
async def test_settlement_queue_discards_open_and_duplicate_updates():
    orch = SimpleNamespace(_settlement_queue=asyncio.Queue(), _settlement_pending_since=0.0)
    await enqueue_contract_settlement(orch, {"proposal_open_contract": {"contract_id": 505, "status": "open"}})
    await enqueue_contract_settlement(orch, {"proposal_open_contract": {"status": "won", "is_settled": 1}})
    assert orch._settlement_queue.empty()
    payload = {"proposal_open_contract": {"contract_id": 505, "status": "won", "is_settled": 1}}
    await asyncio.gather(enqueue_contract_settlement(orch, payload), enqueue_contract_settlement(orch, payload))
    assert orch._settlement_queue.qsize() == 1
    orch._settled_contract_ids = {505}
    await enqueue_contract_settlement(orch, payload)
    assert orch._settlement_queue.qsize() == 1


@pytest.mark.asyncio
async def test_queue_reservation_is_released_when_enqueue_fails():
    queue = SimpleNamespace(put=AsyncMock(side_effect=RuntimeError("queue unavailable")))
    orch = SimpleNamespace(_settlement_queue=queue, _settlement_pending_since=0.0)
    poc = {"contract_id": 508, "status": "won", "is_settled": 1}
    with pytest.raises(RuntimeError, match="queue unavailable"):
        await enqueue_contract_settlement(orch, {"proposal_open_contract": poc})
    assert orch._settlement_queued_ids == set()
