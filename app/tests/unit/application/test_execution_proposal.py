from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.execution_orders import place_order
from src.application.services.orchestrator.execution_proposal import (
    is_proposal_runtime_error,
    is_retriable_proposal_error,
    proposal_retry_scales,
    proposal_stake_attempts,
)
from src.domain.models.trade import TradeDirection, TradeStatus


def test_proposal_stake_attempts_descending():
    attempts = proposal_stake_attempts(100.0, 1.0, [0.85, 0.7, 0.55])
    assert attempts[0] == 100.0
    assert attempts == sorted(attempts, reverse=True)
    assert all(stake >= 1.0 for stake in attempts)


def test_proposal_retry_scales_defaults():
    assert proposal_retry_scales({}) == [0.85, 0.70, 0.55, 0.40]


def test_is_retriable_proposal_error():
    err = RuntimeError("Erro na proposta: Sorry, an error occurred while processing your request.")
    assert is_proposal_runtime_error(err)
    assert is_retriable_proposal_error(err)
    assert not is_retriable_proposal_error(RuntimeError("Erro na proposta: Insufficient balance"))
    assert not is_retriable_proposal_error(ValueError("other"))


def test_proposal_stake_attempts_dedupes_factors():
    attempts = proposal_stake_attempts(10.0, 1.0, [0.5, 0.5, 0.25])
    assert attempts.count(5.0) == 1


def test_proposal_retry_scales_from_config():
    assert proposal_retry_scales({"proposal_retry_scales": [0.9, 0.8]}) == [0.9, 0.8]


@pytest.mark.asyncio
async def test_place_order_retries_lower_stake(orch_config):
    orch = MagicMock()
    orch._active_cycle_id = 14
    orch.config = orch_config
    orch.risk_manager.initial_bankroll = 10000.0
    orch.risk_manager.total_session_profit = 0.0
    orch.risk_manager.contract_to_symbol = {}
    orch.trade_handler.buy_with_parameters = AsyncMock(
        side_effect=[
            RuntimeError("Erro na proposta: Sorry, an error occurred while processing your request."),
            MagicMock(
                contract_id=77,
                payout=90.0,
                buy_price=85.0,
                status=TradeStatus.OPEN,
            ),
        ]
    )
    orch.ws = MagicMock()
    executor = MagicMock()
    executor.orch = orch
    with patch(
        "src.application.services.orchestrator.execution_orders.subscribe_open_contract",
        AsyncMock(),
    ):
        contract = await place_order(executor, "RDBEAR", TradeDirection.CALL, 100.0)
    assert contract.contract_id == 77
    assert orch.trade_handler.buy_with_parameters.await_count == 2


@pytest.mark.asyncio
async def test_place_order_raises_after_all_retries(orch_config):
    orch = MagicMock()
    orch._active_cycle_id = 15
    orch.config = {
        **orch_config,
        "orchestrator": {
            **orch_config["orchestrator"],
            "execution": {"proposal_retry_scales": [0.5]},
        },
    }
    err = RuntimeError("Erro na proposta: Sorry, an error occurred while processing your request.")
    orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=err)
    executor = MagicMock()
    executor.orch = orch
    with (
        patch(
            "src.application.services.orchestrator.execution_orders.subscribe_open_contract",
            AsyncMock(),
        ),
        pytest.raises(RuntimeError, match="Sorry"),
    ):
        await place_order(executor, "RDBEAR", TradeDirection.CALL, 100.0)


@pytest.mark.asyncio
async def test_place_order_raises_when_no_attempts(orch_config):
    orch = MagicMock()
    orch._active_cycle_id = 15
    orch.config = orch_config
    orch.trade_handler.buy_with_parameters = AsyncMock()
    executor = MagicMock()
    executor.orch = orch
    with (
        patch(
            "src.application.services.orchestrator.execution_orders.proposal_stake_attempts",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_orders.subscribe_open_contract",
            AsyncMock(),
        ),
        pytest.raises(RuntimeError, match="falha desconhecida"),
    ):
        await place_order(executor, "RDBEAR", TradeDirection.CALL, 100.0)


@pytest.mark.asyncio
async def test_execute_orders_registers_proposal_skip(orch_config):
    orch = MagicMock()
    orch._active_cycle_id = 16
    orch.config = orch_config
    orch.risk_manager = MagicMock()
    orch.risk_manager.kelly_config = {}
    orch.risk_manager.calculate_stake = MagicMock(return_value=50.0)
    orch.risk_manager.register_entry_conviction = MagicMock()
    orch.risk_manager.record_contract_stake = MagicMock()
    orch.risk_manager.active_contract_ids = []
    orch.state = MagicMock()
    exec_mgr = ExecutionManager(orch)
    exec_mgr._place_order = AsyncMock(
        side_effect=RuntimeError("Erro na proposta: Sorry, an error occurred while processing your request.")
    )
    orders = [("RDBEAR", TradeDirection.CALL, {"execute": True, "trade_score": 0.6})]
    count = await exec_mgr._execute_orders(orders, 0.0, 10000.0)
    assert count == 0
    orch.risk_manager.register_proposal_failure.assert_called_once()
