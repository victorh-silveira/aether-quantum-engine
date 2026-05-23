import pytest

from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState


@pytest.fixture(autouse=True)
def reset_singleton():
    TradingState._instance = None
    yield


@pytest.fixture
def state():
    return TradingState()


@pytest.mark.asyncio
async def test_state_basic_ops(state):
    await state.set_trading(value=True)
    assert state.is_trading is True

    res = await state.finalize_contract(999)
    assert res is None


@pytest.mark.asyncio
async def test_state_serialization(state):
    c = Contract(
        contract_id=1,
        symbol="s",
        direction=TradeDirection.CALL,
        stake=1,
        payout=2,
        status=TradeStatus.OPEN,
        buy_price=1.0,
        proposal_id="p1",
        expiry_time=1700000000,
    )
    await state.add_contract(c)
    state.balance = 500.0

    data = await state.get_state()
    assert data["balance"] == 500.0
    assert "1" in data["active_contracts"]
    assert data["active_contracts"]["1"]["status"] == "OPEN"


@pytest.mark.asyncio
async def test_trading_state_summary():
    state = TradingState()
    state.balance = 1000.0
    summary = await state.get_state()
    assert summary["balance"] == 1000.0
    assert "active_contracts" in summary


@pytest.mark.asyncio
async def test_trading_state_contract_edge_cases():
    state = TradingState()
    await state.finalize_contract(999)
    assert 999 not in state.active_contracts
