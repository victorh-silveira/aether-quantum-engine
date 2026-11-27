"""Testes unitarios para fechamento de 100% em execution e settlement_outcome."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_contract_registration import register_confirmed_contract
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.execution_orders import _subscribe_open_contract_background, place_order
from src.application.services.orchestrator.execution_proposal import (
    is_retriable_proposal_error,
    proposal_retry_scales,
    proposal_stake_attempts,
)
from src.application.services.orchestrator.execution_settlement import (
    _handle_stagnant_settlement,
    _next_stagnant_poll_count,
    _prune_orphan_settlement_ids,
    _settlement_grace_period,
    _settlement_poll_delay,
    _settlement_timed_out,
    _wait_broker_offline_settlement,
    reconcile_contracts,
    run_settlement_watch,
    wait_for_settlement,
)
from src.application.services.orchestrator.settlement_outcome import (
    check_session_limits_before_post_settlement,
    process_contract_outcome,
    sync_state_manager_session,
)
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.state.state_manager import StateManager
from src.infrastructure.state.trading_state import TradingState


def _make_orch():
    TradingState.reset()
    state = TradingState()
    state.balance = 1000.0
    return SimpleNamespace(
        config={
            "laya": {"stake_min": 1.0},
            "risk_management": {"params": {"duration": 5, "duration_unit": "m", "stake_min": 1.0}},
            "orchestrator": {"execution": {"settlement_poll_seconds": 0.0, "settlement_max_stagnant_polls": 1}},
        },
        state=state,
        risk_manager=RiskManager({"params": {"stake_min": 1.0}}),
        state_mgr=StateManager(),
        ws=MagicMock(is_running=True),
        auth=MagicMock(),
        trade_handler=MagicMock(),
        business_instrumentor=MagicMock(runtime=MagicMock()),
        logger=MagicMock(),
        running=True,
        _active_cycle_id=1,
        _contract_cycle={},
        _cluster_results=[],
        _session_wins=0,
        _session_losses=0,
        _buffer_result_logs=False,
        _pending_result_logs=[],
        trading_transport="ws",
        deriv_account_id="VRTC1",
        save_full_state=AsyncMock(),
        schedule_trading_cycle_after_settlement=MagicMock(),
    )


@pytest.mark.asyncio
async def test_execution_manager_laya_order_and_proposal_helpers():
    assert is_retriable_proposal_error(ValueError("x")) is False
    assert is_retriable_proposal_error(RuntimeError("Erro na proposta: temporary")) is True
    assert is_retriable_proposal_error(RuntimeError("Erro na proposta: insufficient balance")) is False
    assert proposal_stake_attempts(10.0, 1.0, None)[0] == 10.0
    assert proposal_retry_scales({"proposal_retry_scales": [0.5]}) == [0.5]
    assert proposal_retry_scales({}) == [0.85, 0.70, 0.55, 0.40]

    orch = _make_orch()
    mgr = ExecutionManager(orch)
    dec = LayaDecision("R_10", TradeDirection.CALL, 0.80, 0.80, 0.20, 0.80, 10.0, {"ema_20": 1.0})
    contract = Contract(101, "p1", TradeStatus.OPEN, 10.0, 19.5, "R_10", TradeDirection.CALL, 10.0, 9999999999)

    with (
        patch.object(mgr, "_place_order", AsyncMock(return_value=None)),
        patch.object(mgr, "_flush_result_buffer"),
    ):
        assert await mgr.execute_laya_order(dec, 10.0, 300) is False

    with (
        patch.object(mgr, "_place_order", AsyncMock(return_value=contract)),
        patch.object(mgr, "_run_settlement_watch", AsyncMock()),
        patch.object(mgr, "_flush_result_buffer"),
    ):
        assert await mgr.execute_laya_order(dec, 10.0, 300) is True
        # Segunda tentativa no mesmo contrato ja ativo retorna True no register_confirmed_contract
        assert await register_confirmed_contract(orch, contract, "R_10", 10.0) is True
        orch._settled_contract_ids = {101}
        assert await register_confirmed_contract(orch, contract, "R_10", 10.0) is False


@pytest.mark.asyncio
async def test_execution_orders_ws_rest_and_retry_paths():
    orch = _make_orch()
    mgr = ExecutionManager(orch)
    contract = Contract(202, "p2", TradeStatus.OPEN, 10.0, 19.5, "R_10", TradeDirection.PUT, 10.0, 9999999999)

    # subscribe background exception branch
    with patch(
        "src.application.services.orchestrator.execution_orders.subscribe_open_contract",
        AsyncMock(side_effect=RuntimeError("sub fail")),
    ):
        await _subscribe_open_contract_background(orch.ws, 202, timeout=1.0, cid="C0001")

    # Retry em proposal error retriable e falha final
    orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=[RuntimeError("Erro na proposta: glitch"), contract])
    with patch("src.application.services.orchestrator.execution_orders.subscribe_open_contract", AsyncMock()):
        res = await place_order(mgr, "R_10", TradeDirection.PUT, 10.0, metrics={"_order_candle_epoch": 300})
    assert res is contract

    orch.trade_handler.buy_with_parameters = AsyncMock(side_effect=RuntimeError("API maintenance"))
    with (
        patch(
            "src.application.services.orchestrator.execution_orders.handle_broker_maintenance_error", return_value=True
        ),
        pytest.raises(RuntimeError, match="maintenance"),
    ):
        await place_order(mgr, "R_10", TradeDirection.PUT, 10.0)

    # REST transport path + rest balance refresh error
    orch.trading_transport = "rest"
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=contract)
    orch.auth.rest_client.return_value = SimpleNamespace(
        list_accounts=AsyncMock(return_value=[SimpleNamespace(account_id="VRTC1", balance=990.0)])
    )
    with patch(
        "src.application.services.orchestrator.execution_orders.schedule_rest_contract_settlement"
    ) as sched_rest:
        res_rest = await place_order(mgr, "R_10", TradeDirection.CALL, 10.0)
    assert res_rest is contract
    sched_rest.assert_called_once()

    orch.auth.rest_client.side_effect = RuntimeError("rest down")
    with patch(
        "src.application.services.orchestrator.execution_orders.schedule_rest_contract_settlement"
    ) as sched_rest2:
        await place_order(mgr, "R_10", TradeDirection.CALL, 10.0)
    sched_rest2.assert_called_once()


@pytest.mark.asyncio
async def test_execution_settlement_and_outcome_branches():
    orch = _make_orch()
    mgr = ExecutionManager(orch)
    contract = Contract(303, "p3", TradeStatus.OPEN, 10.0, 19.5, "R_10", TradeDirection.CALL, 10.0, 9999999999)
    await orch.state.add_contract(contract)
    orch.risk_manager.register_open_contract(303, "R_10", 10.0)

    await _settlement_poll_delay(0.0)
    assert _settlement_grace_period(mgr, {}, 0.0) >= 0.0
    assert _settlement_timed_out(mgr, 0.0, 1) is True
    assert orch.risk_manager.active_contract_ids == []

    orch.risk_manager.register_open_contract(999, "R_10", 10.0)
    assert _prune_orphan_settlement_ids(mgr) is True

    orch.ws.is_running = False
    assert await _wait_broker_offline_settlement(mgr, 0.0) is True
    orch.ws.is_running = True
    assert await _wait_broker_offline_settlement(mgr, 0.0) is False

    assert _next_stagnant_poll_count(0, 1.0, 10.0, [1], [1]) == 0
    assert _next_stagnant_poll_count(1, 20.0, 10.0, [1], [1]) == 2
    assert _next_stagnant_poll_count(1, 20.0, 10.0, [1], [2]) == 0

    assert await _handle_stagnant_settlement(mgr, 0.0, 5, 1) is None
    orch.ws.is_running = False
    assert await _handle_stagnant_settlement(mgr, 0.0, 1, 1) == "continue"
    orch.ws.is_running = True
    orch.risk_manager.register_open_contract(303, "R_10", 10.0)
    with patch(
        "src.application.services.orchestrator.execution_settlement.backfill_pending_contracts",
        AsyncMock(return_value=1),
    ):
        assert await _handle_stagnant_settlement(mgr, 0.0, 1, 1) == "continue"
    orch.risk_manager.active_contract_ids = []
    assert await _handle_stagnant_settlement(mgr, 0.0, 1, 1) == "break"

    # reconcile_contracts branches
    orch.ws.is_running = False
    assert await reconcile_contracts(mgr) is False
    orch.ws.is_running = True
    with patch("src.application.services.orchestrator.execution_settlement.reconcile_single_contract", AsyncMock()):
        assert await reconcile_contracts(mgr) is True
    with patch(
        "src.application.services.orchestrator.execution_settlement.reconcile_single_contract",
        AsyncMock(side_effect=TimeoutError("t")),
    ):
        assert await reconcile_contracts(mgr) is False
    orch.ws.is_running = True
    with patch(
        "src.application.services.orchestrator.execution_settlement.reconcile_single_contract",
        AsyncMock(side_effect=ValueError("v")),
    ):
        assert await reconcile_contracts(mgr) is True

    # wait_for_settlement & run_settlement_watch
    orch.risk_manager.register_open_contract(303, "R_10", 10.0)
    with (
        patch.object(mgr, "reconcile_contracts", AsyncMock(side_effect=[False, True])),
        patch(
            "src.application.services.orchestrator.execution_settlement._handle_stagnant_settlement",
            AsyncMock(return_value="break"),
        ),
        patch("src.application.services.orchestrator.execution_settlement._settlement_poll_delay", AsyncMock()),
    ):
        await wait_for_settlement(mgr, timeout=3600)
    orch.state.active_contracts.clear()
    with patch(
        "src.application.services.orchestrator.execution_settlement.wait_for_settlement",
        AsyncMock(side_effect=RuntimeError("err")),
    ):
        await run_settlement_watch(mgr)
    orch.schedule_trading_cycle_after_settlement.assert_called_once()

    # settlement_outcome
    orch.risk_manager.register_open_contract(303, "R_10", 10.0)
    orch.business_metrics = MagicMock()
    called_summary = MagicMock()
    process_contract_outcome(orch, {"balance_after": 1009.5}, contract, 303, 9.5, log_cluster_summary=called_summary)
    called_summary.assert_called_once()
    assert orch._session_wins == 1
    # Aceita balance_after real da Deriv mesmo com salto > $2.00 (ex.: contrato externo liquidado)
    process_contract_outcome(orch, {"balance_after": 1677.08}, contract, 305, 10.0)
    assert orch.state.balance == pytest.approx(1677.08)
    process_contract_outcome(orch, {}, contract, 304, -10.0)
    assert orch._session_losses == 1
    assert sync_state_manager_session(orch) is False
    assert sync_state_manager_session(SimpleNamespace()) is False
    assert check_session_limits_before_post_settlement(orch) is False

    # Stop-win atingido com StateManager real baseado no lucro realizado da sessao
    mgr = StateManager()
    mgr.persistence = SimpleNamespace(load=lambda: None, save=lambda _d: None)
    mgr.reset_session_metrics(100.0, 5.0)
    orch_sw = SimpleNamespace(
        state_mgr=mgr,
        state=SimpleNamespace(balance=800.0),
        risk_manager=SimpleNamespace(initial_bankroll=100.0, total_session_profit=6.0),
        logger=MagicMock(),
        _session_wins=2,
        _session_losses=1,
    )
    assert sync_state_manager_session(orch_sw, target=5.0) is True
    assert check_session_limits_before_post_settlement(orch_sw) is True
    assert any("SESSION_STATS |" in call.args[0] for call in orch_sw.logger.info.call_args_list)
    assert any("STOP_WIN |" in call.args[0] for call in orch_sw.logger.info.call_args_list)
