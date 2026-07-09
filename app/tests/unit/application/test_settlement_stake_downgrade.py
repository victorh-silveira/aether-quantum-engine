from unittest.mock import AsyncMock

import pytest

from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


@pytest.mark.asyncio
async def test_settlement_loss_reconciles_planned_vs_executed_stake(orch_ready):
    orch = orch_ready
    orch._contract_cycle = {901: 4}
    contract = Contract(
        contract_id=901,
        proposal_id="p901",
        status=TradeStatus.OPEN,
        buy_price=332.28,
        payout=610.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=390.92,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [901]
    orch.risk_manager.contract_to_symbol[901] = "RDBULL"
    orch.risk_manager.contract_stakes[901] = 390.92
    orch.risk_manager.begin_cluster(1)
    data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 901,
            "buy_price": 332.28,
            "profit": -390.92,
            "balance_after": 10667.72,
        }
    }
    with patch_instant_post_settlement_poll():
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    assert orch.risk_manager.pending_loss.get("RDBULL") == pytest.approx(390.92)
    assert orch.risk_manager.last_loss_stake == pytest.approx(332.28)
    assert orch.risk_manager.total_session_profit == pytest.approx(-332.28)


@pytest.mark.asyncio
async def test_process_contract_settlement_win_with_stake_downgrade_retains_pending_loss(orch_ready):
    orch = orch_ready
    orch._contract_cycle = {808: 66}
    orch.risk_manager.pending_loss["RDBULL"] = 349.81
    orch.risk_manager.contract_requested_stakes[808] = 349.81
    contract = Contract(
        contract_id=808,
        proposal_id="p808",
        status=TradeStatus.OPEN,
        buy_price=297.34,
        payout=540.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=349.81,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [808]
    orch.risk_manager.contract_to_symbol[808] = "RDBULL"
    orch.risk_manager.contract_stakes[808] = 349.81
    orch.risk_manager.begin_cluster(1)
    data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 808,
            "buy_price": 297.34,
            "profit": 85.0,
            "balance_after": 1085.0,
        }
    }
    with patch_instant_post_settlement_poll():
        await process_contract_settlement(orch, data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    assert orch.risk_manager.pending_loss.get("RDBULL") == pytest.approx(317.28)
    assert sum(orch.risk_manager.pending_loss.values()) > 0.0
