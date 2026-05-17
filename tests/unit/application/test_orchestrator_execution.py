import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_execute_cluster_dispatches_when_decisions_present(orch_config):
    TradingState.reset()
    oe = orch_config.setdefault("orchestrator", {})
    ex = oe.setdefault("execution", {})
    ex["include_anchor_trades"] = True
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.balance = 1000.0
        decisions = {
            "1HZ75V": {
                "direction": TradeDirection.CALL,
                "metrics": {"conviction": 1.0, "macro_bias": 0.8, "pattern_tags": ["BULL_FLAG"]},
            },
            "R_50": {
                "direction": TradeDirection.CALL,
                "metrics": {"conviction": 0.9, "macro_bias": 0.4, "pattern_tags": ["BULL_PENNANT"]},
            },
        }

        async def _place_order_with_buffer(symbol, direction, stake, **_kw):
            orch._pending_result_logs = ["   | RESULT: 1HZ75V  | CALL | WIN  | P&L: $+1.00 | api=won"]
            return Contract(
                contract_id=1,
                proposal_id="p1",
                status=TradeStatus.OPEN,
                buy_price=1.0,
                payout=2.0,
                symbol="1HZ75V",
                direction=TradeDirection.CALL,
                stake=1.0,
                expiry_time=0,
            )

        orch.executor._place_order = AsyncMock(side_effect=_place_order_with_buffer)
        orch.executor.wait_for_settlement = AsyncMock()
        await orch.executor.execute_cluster(decisions)
        assert orch.executor._place_order.call_count >= 1


@pytest.mark.asyncio
async def test_contract_update_won(orch_config):
    """Testa a atualizacao de contrato vitorioso."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.active_contracts[1] = Contract(
            contract_id=1,
            proposal_id="p1",
            status=TradeStatus.OPEN,
            buy_price=10.0,
            payout=18.0,
            symbol="1HZ75V",
            direction=TradeDirection.PUT,
            stake=10.0,
            expiry_time=0,
        )
        orch.risk_manager.contract_to_symbol[1] = "1HZ75V"
        data = {
            "proposal_open_contract": {
                "contract_id": 1,
                "is_settled": 1,
                "status": "won",
                "profit": 10.0,
                "balance_after": 1010.0,
            }
        }
        await orch._on_contract_update(data)
        assert orch.state.balance == 1010.0


@pytest.mark.asyncio
async def test_execution_manager_skip_and_failure_paths(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.balance = 1000.0
        orch.symbols = ["1HZ75V", "R_50"]
        decisions = {
            "1HZ75V": {"direction": None, "metrics": {"conviction": 0.0}},
            "R_50": {
                "direction": TradeDirection.CALL,
                "metrics": {"conviction": 1.0},
            },
        }
        orch.executor._place_order = AsyncMock(side_effect=Exception("API ERROR"))
        orch.executor.wait_for_settlement = AsyncMock()
        await orch.executor.execute_cluster(decisions)
        assert orch.executor._place_order.call_count == 1


@pytest.mark.asyncio
async def test_wait_for_settlement_polls_reconcile(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        c = Contract(
            contract_id=1,
            symbol="1HZ75V",
            direction=TradeDirection.CALL,
            stake=10.0,
            payout=18.0,
            status=TradeStatus.OPEN,
            buy_price=10.0,
            proposal_id="p1",
            expiry_time=int(time.time()) + 600,
        )
        await orch.state.add_contract(c)
        orch.risk_manager.active_contract_ids = [1]
        n = [0]

        async def mock_reconcile():
            n[0] += 1
            if n[0] >= 1:
                orch.risk_manager.active_contract_ids = []

        with (
            patch.object(orch.executor, "reconcile", side_effect=mock_reconcile),
            patch("src.application.services.orchestrator.execution_manager.asyncio.sleep", AsyncMock()),
        ):
            await orch.executor.wait_for_settlement(timeout=300)
        assert n[0] >= 1


@pytest.mark.asyncio
async def test_execution_manager_inter_symbol_delay(orch_config):
    TradingState.reset()
    orch_config["symbols"] = ["1HZ75V", "R_50"]
    orch_config["orchestrator"]["execution"]["inter_symbol_delay"] = 0.25
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.balance = 1000.0
        decisions = {
            "1HZ75V": {"direction": TradeDirection.CALL, "metrics": {"conviction": 1.0}},
            "R_50": {"direction": TradeDirection.PUT, "metrics": {"conviction": 1.0}},
        }
        orch.executor._place_order = AsyncMock(return_value=MagicMock(contract_id=1))
        orch.executor.wait_for_settlement = AsyncMock()
        with patch("src.application.services.orchestrator.execution_manager.asyncio.sleep", AsyncMock()) as mock_sleep:
            await orch.executor.execute_cluster(decisions)
        mock_sleep.assert_awaited_with(0.25)


@pytest.mark.asyncio
async def test_execution_manager_multiplier_tp_calculation(orch_config):
    """Verifica o cálculo de TP para atingir a meta de 3%."""
    TradingState.reset()
    orch_config["risk_management"]["params"]["contract_type"] = "MULTIPLIER"
    orch_config["risk_management"]["large_account_stop_win_pct"] = 3.0
    orch_config["risk_management"]["small_account_threshold"] = 0.0

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.state.balance = 1000.0
        orch.risk_manager.set_initial_bankroll(1000.0)
        orch.risk_manager.total_session_profit = 10.0  # Already made $10, need $20 more

        # Mock get_proposal to check params
        orch.trade_handler.get_proposal = AsyncMock()
        orch.trade_handler.buy_contract = AsyncMock(return_value=MagicMock(contract_id=123))

        await orch.executor._place_order("frxEURUSD", TradeDirection.CALL, 50.0)

        # Check if take_profit was set to $20.0
        args, kwargs = orch.trade_handler.get_proposal.call_args
        params = kwargs.get("params") or args[3]
        assert params["limit_order"]["take_profit"] == 20.0
