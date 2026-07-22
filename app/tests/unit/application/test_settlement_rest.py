from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_rest import (
    _synthetic_poc,
    schedule_rest_contract_settlement,
    settle_rest_contract_when_due,
)
from src.domain.models.trade import Contract, TradeDirection, TradeStatus


def _contract(cid: int = 1) -> Contract:
    return Contract(
        contract_id=cid,
        proposal_id="t",
        status=TradeStatus.OPEN,
        buy_price=0.35,
        payout=0.66,
        symbol="R_10",
        direction=TradeDirection.CALL,
        stake=0.35,
        expiry_time=0,
        longcode="x",
    )


def test_synthetic_poc_win_and_loss():
    won = _synthetic_poc(contract_id=1, won=True, buy_price=0.35, payout=0.66)
    lost = _synthetic_poc(contract_id=2, won=False, buy_price=0.35, payout=0.66)
    assert won["proposal_open_contract"]["status"] == "won"
    assert lost["proposal_open_contract"]["profit"] == pytest.approx(-0.35)


@pytest.mark.asyncio
async def test_settle_rest_contract_when_due_marks_win():
    orch = MagicMock()
    orch.deriv_account_id = "DOT1"
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(
        return_value=[MagicMock(account_id="DOT1", balance=110.0)]
    )
    with (
        patch("src.application.services.orchestrator.settlement_rest.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "src.application.services.orchestrator.settlement_rest.process_contract_settlement",
            new_callable=AsyncMock,
        ) as settle,
    ):
        await settle_rest_contract_when_due(orch, _contract(9), balance_after_buy=100.0)
    settle.assert_awaited_once()
    payload = settle.await_args.args[1]
    assert payload["proposal_open_contract"]["status"] == "won"


@pytest.mark.asyncio
async def test_settle_rest_uses_first_account_when_id_mismatch():
    orch = MagicMock()
    orch.deriv_account_id = "OTHER"
    orch.state.balance = 1.0
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(
        return_value=[MagicMock(account_id="DOT1", balance=42.0)]
    )
    with (
        patch("src.application.services.orchestrator.settlement_rest.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "src.application.services.orchestrator.settlement_rest.process_contract_settlement",
            new_callable=AsyncMock,
        ),
    ):
        await settle_rest_contract_when_due(orch, _contract(1), balance_after_buy=100.0)


@pytest.mark.asyncio
async def test_settle_rest_empty_accounts_uses_state_balance():
    orch = MagicMock()
    orch.deriv_account_id = "DOT1"
    orch.state.balance = 77.0
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(return_value=[])
    with (
        patch("src.application.services.orchestrator.settlement_rest.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "src.application.services.orchestrator.settlement_rest.process_contract_settlement",
            new_callable=AsyncMock,
        ) as settle,
    ):
        await settle_rest_contract_when_due(orch, _contract(8), balance_after_buy=100.0)
    payload = settle.await_args.args[1]
    assert payload["proposal_open_contract"]["status"] == "lost"


@pytest.mark.asyncio
async def test_settle_rest_handles_failure():
    orch = MagicMock()
    with patch(
        "src.application.services.orchestrator.settlement_rest._balance_after_wait",
        AsyncMock(side_effect=RuntimeError("down")),
    ):
        await settle_rest_contract_when_due(orch, _contract(2), balance_after_buy=100.0)


@pytest.mark.asyncio
async def test_schedule_rest_contract_settlement_tracks_task():
    orch = MagicMock()
    orch._rest_settlement_tasks = None
    with patch(
        "src.application.services.orchestrator.settlement_rest.settle_rest_contract_when_due",
        new_callable=AsyncMock,
    ):
        schedule_rest_contract_settlement(orch, _contract(3), balance_after_buy=50.0)
        assert isinstance(orch._rest_settlement_tasks, set)
        assert len(orch._rest_settlement_tasks) == 1
        task = next(iter(orch._rest_settlement_tasks))
        await task
