"""Fluxo M5 de stake e reserva sem compras reais."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.domain.models.trade import TradeDirection


def _executor(pending):
    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {"R_25": pending} if pending else {}
    risk.pending_loss_total.return_value = pending
    risk.active_contract_ids = set()
    orch = SimpleNamespace(
        config={
            "trading": {"mode": "demo"},
            "risk_management": {"params": {"stake_min": 1, "duration": 5}},
            "orchestrator": {"execution": {"multi_symbol_orders": True}},
            "deep_learning": {},
        },
        risk_manager=risk,
        state=SimpleNamespace(balance=1000.0),
        _active_cycle_id=1,
        business_metrics=None,
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(return_value=None)
    return executor


def _order(symbol):
    return (
        symbol,
        TradeDirection.CALL,
        {
            "direction_origin": "PULLBACK_SETUP",
            "indicator_controls_execution": True,
            "indicator_quote_rate": 0.8,
            "indicator_period_seconds": 300,
            "indicator_model_version": "pullback-1-v2",
        },
    )


@pytest.mark.asyncio
async def test_pullback_base_and_recovery_reserve_by_account():
    executor = _executor(100)
    executor._place_order.side_effect = [SimpleNamespace(contract_id=7, buy_price=50), None]
    orders = [_order("R_25"), _order("R_10")]
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.claim_indicator_order_once",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.release_unbought_order_claim",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.execution_manager_execute.consensus_enabled", return_value=True),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.register_confirmed_contract",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.record_model_attribution",
            new_callable=AsyncMock,
        ),
    ):
        assert await execute_cluster_orders(executor, orders, 0, 1000) == 1
    first = executor._place_order.await_args_list[0]
    second = executor._place_order.await_args_list[1]
    assert first.args[2] == 50
    assert first.kwargs["metrics"]["indicator_stake_mode"] == "PULLBACK_RECOVERY"
    assert second.args[2] == 9.5
    assert second.kwargs["metrics"]["indicator_stake_mode"] == "PULLBACK_BASE"
    assert 7 in executor.orch._pullback_recovery_claims


@pytest.mark.asyncio
async def test_pullback_no_loss_uses_one_percent():
    executor = _executor(0)
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.claim_indicator_order_once",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.release_unbought_order_claim",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.execution_manager_execute.consensus_enabled", return_value=True),
    ):
        await execute_cluster_orders(executor, [_order("R_25")], 0, 1000)
    assert executor._place_order.await_args.args[2] == 10


@pytest.mark.asyncio
async def test_pullback_does_not_buy_if_fallback_cannot_cover_minimum():
    executor = _executor(100)
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
        ),
        patch("src.application.services.orchestrator.execution_manager_execute.consensus_enabled", return_value=True),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.claim_recovery",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.pullback_stake",
            side_effect=[(50, 40), (0, 0)],
        ),
    ):
        assert await execute_cluster_orders(executor, [_order("R_25")], 0, 1000) == 0
    executor._place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_pullback_single_strike_sizing_when_enabled():
    executor = _executor(0)
    executor.orch.config["pullback_strategy"] = {"enabled": True, "single_strike": True}
    executor.orch.config["risk_management"]["params"]["compounding_rate_daily"] = 0.015
    executor.orch.config["risk_management"]["params"]["compounding_enabled"] = True
    executor.orch.risk_manager.initial_bankroll = 1000.0
    executor.orch.risk_manager.total_session_profit = 0.0
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.claim_indicator_order_once",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.release_unbought_order_claim",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.execution_manager_execute.consensus_enabled", return_value=True),
    ):
        await execute_cluster_orders(executor, [_order("R_25")], 0, 1000)
    assert executor._place_order.await_args.args[2] == 18.75


@pytest.mark.asyncio
async def test_execution_manager_quarantines_symbol_on_closed_session():
    executor = _executor(0)
    executor.orch.config["pullback_strategy"] = {"enabled": True, "dynamic_rotation": True}
    executor.orch._symbol_loss_quarantine = {}
    executor._place_order.side_effect = RuntimeError("market is closed")
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.signal_epoch_current", return_value=True
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.claim_indicator_order_once",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.release_unbought_order_claim",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.execution_manager_execute.consensus_enabled", return_value=True),
    ):
        await execute_cluster_orders(executor, [_order("R_25")], 0, 1000)
    assert "R_25" in executor.orch._symbol_loss_quarantine
