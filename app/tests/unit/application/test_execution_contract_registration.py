"""Registro de compra contra callbacks concorrentes de liquidacao."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services import indicator_consensus
from src.application.services.orchestrator.execution_contract_registration import register_confirmed_contract
from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.application.services.orchestrator.settlement_dedupe import settlement_lock
from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.trade_handler_errors import PurchaseOutcomeUnknownError
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_concurrent_registration_of_same_contract_is_idempotent():
    TradingState.reset()
    risk = MagicMock()
    risk.active_contract_ids = []
    risk.contract_to_symbol = {}
    orch = SimpleNamespace(state=TradingState(), risk_manager=risk, _contract_cycle={}, _active_cycle_id=4)
    contract = SimpleNamespace(contract_id=601)
    assert await asyncio.gather(
        register_confirmed_contract(orch, contract, "R_10", 10.0),
        register_confirmed_contract(orch, contract, "R_10", 10.0),
    ) == [True, True]
    risk.record_contract_stake.assert_called_once_with(601, 10.0)
    assert risk.active_contract_ids == [601]
    assert orch.state.active_contracts == {601: contract}


@pytest.mark.asyncio
async def test_fast_settlement_cannot_resurrect_contract():
    TradingState.reset()
    risk = MagicMock()
    risk.active_contract_ids = []
    risk.contract_to_symbol = {}
    orch = SimpleNamespace(state=TradingState(), risk_manager=risk, _contract_cycle={}, _active_cycle_id=4)
    contract = SimpleNamespace(contract_id=602)
    async with settlement_lock(orch):
        orch._settled_contract_ids = {602}
    assert not await register_confirmed_contract(orch, contract, "R_25", 10.0)
    assert not orch.state.active_contracts
    risk.record_contract_stake.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_buy_outcome_quarantines_symbol_before_next_cycle():
    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    risk.calculate_stake.return_value = 10.0
    orch = SimpleNamespace(
        config={"deep_learning": {}, "risk_management": {"params": {"duration": 300}}},
        risk_manager=risk,
        state=SimpleNamespace(balance=1000.0),
        _active_cycle_id=1,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(side_effect=PurchaseOutcomeUnknownError("no ACK"))
    order = [("R_10", TradeDirection.CALL, {})]
    assert await execute_cluster_orders(executor, order, 0.0, 1000.0) == 0
    assert "R_10" in orch._unknown_purchase_until
    assert await execute_cluster_orders(executor, order, 0.0, 1000.0) == 0
    executor._place_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejected_quote_allows_new_proposal_in_same_candle(monkeypatch):
    monkeypatch.setattr(indicator_consensus.time, "time", lambda: 620)
    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    orch = SimpleNamespace(
        config={
            "indicator_consensus": {"enabled": True, "stake_pct": 0.01},
            "deep_learning": {},
            "risk_management": {"params": {"duration": 300}},
        },
        risk_manager=risk,
        state=SimpleNamespace(balance=1000.0),
        _active_cycle_id=1,
        state_store=None,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(return_value=None)
    order = [("R_25", TradeDirection.CALL, {"indicator_controls_execution": True, "indicator_candle_epoch": 300})]
    assert await execute_cluster_orders(executor, order, 0.0, 1000.0) == 0
    assert await execute_cluster_orders(executor, order, 0.0, 1000.0) == 0
    assert executor._place_order.await_count == 2
