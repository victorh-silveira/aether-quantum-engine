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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction,p_call,rate",
    [
        (TradeDirection.CALL, 0.62, 0.80),
        (TradeDirection.PUT, 0.38, 0.80),
        (TradeDirection.CALL, 0.52, 0.80),
        (TradeDirection.PUT, 0.62, 0.80),
        (TradeDirection.CALL, 0.62, None),
    ],
)
async def test_order_does_not_veto_negative_quote_edge(orch_config, direction, p_call, rate):
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = orch_config
    orch.risk_manager.contract_to_symbol = {}
    orch.trade_handler.fetch_proposal_payout = AsyncMock(return_value=rate)
    orch.trade_handler.buy_with_parameters = AsyncMock(
        return_value=MagicMock(contract_id=11, buy_price=10.0, payout=18.0)
    )
    executor = MagicMock(orch=orch)
    metrics = {"calibrated_prob": p_call}
    with patch("src.application.services.orchestrator.execution_orders.subscribe_open_contract", AsyncMock()):
        await place_order(executor, "1HZ75V", direction, 10.0, metrics=metrics)
    orch.trade_handler.buy_with_parameters.assert_awaited_once()
    orch.trade_handler.fetch_proposal_payout.assert_not_awaited()
    assert metrics["quote_edge"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("direction,p_call", [(TradeDirection.CALL, 0.57), (TradeDirection.PUT, 0.43)])
async def test_order_haircut_reports_thin_quote_without_veto(orch_config, direction, p_call):
    orch = MagicMock()
    orch.config = orch_config
    orch.config["orchestrator"]["execution"].update({"quote_probability_haircut": 0.02})
    orch.trade_handler.fetch_proposal_payout = AsyncMock(return_value=0.8)
    orch.trade_handler.buy_with_parameters = AsyncMock(
        return_value=MagicMock(contract_id=12, buy_price=10.0, payout=18.0)
    )
    metrics = {"calibrated_prob": p_call}
    with patch("src.application.services.orchestrator.execution_orders.subscribe_open_contract", AsyncMock()):
        await place_order(MagicMock(orch=orch), "1HZ75V", direction, 10.0, metrics=metrics)
    assert metrics["quote_edge"] < 0.0
    orch.trade_handler.buy_with_parameters.assert_awaited_once()


@pytest.mark.asyncio
async def test_quote_guard_does_not_require_prequote_for_rest(orch_config):
    orch = MagicMock()
    orch.config = orch_config
    orch.trade_handler.trading_transport = "rest"
    orch.trade_handler.fetch_proposal_payout = AsyncMock(return_value=0.8)
    orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=RuntimeError("rest attempted"))
    with pytest.raises(RuntimeError, match="rest attempted"):
        await place_order(MagicMock(orch=orch), "1HZ75V", TradeDirection.CALL, 10.0, metrics={"calibrated_prob": 0.65})
    orch.trade_handler.buy_with_parameters.assert_awaited()
    orch.trade_handler.fetch_proposal_payout.assert_not_awaited()


def test_proposal_stake_attempts_descending():
    attempts = proposal_stake_attempts(100.0, 1.0, [0.85, 0.7, 0.55])
    assert attempts[0] == 100.0
    assert attempts == sorted(attempts, reverse=True)
    assert all(stake >= 1.0 for stake in attempts)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["demo", "live"])
async def test_purchase_respects_absolute_cap(orch_config, mode):
    """A compra respeita teto absoluto mesmo com stake calculada maior."""
    orch = MagicMock()
    orch.config = orch_config
    orch.config["risk_management"]["kelly"] = {"max_stake": 15.0}
    orch.auth.mode = mode
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=MagicMock(contract_id=1))
    with patch("src.application.services.orchestrator.execution_orders.subscribe_open_contract", AsyncMock()):
        await place_order(MagicMock(orch=orch), "1HZ75V", TradeDirection.CALL, 100.0)
    assert orch.trade_handler.buy_with_parameters.await_args.args[2] == 15.0


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
        contract = await place_order(executor, "R_10", TradeDirection.CALL, 100.0)
    assert contract.contract_id == 77
    assert orch.trade_handler.buy_with_parameters.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["demo", "live"])
async def test_unqualified_checkpoint_order_never_exceeds_cap(orch_config, mode):
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.config = orch_config
    orch.auth.mode = mode
    orch.state.balance = 1000.0
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=MagicMock(contract_id=1))
    executor = MagicMock(orch=orch)
    metrics = {"checkpoint_exploration": True, "provisional_max_stake_pct": 0.001}
    with patch("src.application.services.orchestrator.execution_orders.subscribe_open_contract", AsyncMock()):
        await place_order(executor, "R_10", TradeDirection.CALL, 10.0, metrics=metrics)
    assert orch.trade_handler.buy_with_parameters.await_args.args[2] <= 1.0


@pytest.mark.asyncio
async def test_unqualified_checkpoint_rejects_minimum_above_cap(orch_config):
    orch = MagicMock()
    orch.config = orch_config
    orch.state.balance = 100.0
    executor = MagicMock(orch=orch)
    with pytest.raises(RuntimeError, match="stake minimo excede teto"):
        await place_order(
            executor,
            "R_10",
            TradeDirection.CALL,
            10.0,
            metrics={"checkpoint_exploration": True, "provisional_max_stake_pct": 0.001},
        )


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
        await place_order(executor, "R_10", TradeDirection.CALL, 100.0)


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
        await place_order(executor, "R_10", TradeDirection.CALL, 100.0)


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
    orders = [("R_10", TradeDirection.CALL, {"execute": True, "trade_score": 0.6})]
    count = await exec_mgr._execute_orders(orders, 0.0, 10000.0)
    assert count == 0
    orch.risk_manager.register_proposal_failure.assert_called_once()


@pytest.mark.asyncio
async def test_place_order_blocks_when_quote_loses_edge(orch_config):
    orch = MagicMock()
    orch._active_cycle_id = 17
    orch.config = orch_config
    orch.risk_manager.contract_to_symbol = {}
    orch.trade_handler.buy_with_parameters = AsyncMock(
        side_effect=RuntimeError("Cotacao final Rise/Fall perdeu vantagem; compra bloqueada")
    )
    executor = MagicMock(orch=orch)
    res = await place_order(executor, "1HZ75V", TradeDirection.CALL, 10.0, metrics={"calibrated_prob": 0.45})
    assert res is None


def test_attach_quote_guard_params_disabled_when_require_quote_edge_false():
    from src.application.services.orchestrator.execution_orders import _attach_quote_guard_params

    params = {"duration": 5}
    metrics = {"calibrated_prob": 0.55}
    _attach_quote_guard_params(params, metrics, TradeDirection.CALL, exec_cfg={"require_quote_edge": False})
    assert "_quote_guard_side_probability" not in params
    assert "_quote_guard_min_edge" not in params
    assert "min_payout_rate" not in params

    _attach_quote_guard_params(params, metrics, TradeDirection.CALL, exec_cfg={"require_quote_edge": True})
    assert "_quote_guard_side_probability" in params
    assert "_quote_guard_min_edge" in params
    assert "min_payout_rate" in params
