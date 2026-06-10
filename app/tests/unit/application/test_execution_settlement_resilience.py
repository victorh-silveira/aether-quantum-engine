from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_settlement import reconcile_contracts
from src.application.services.orchestrator.settlement_utils import is_transient_broker_error, mark_ws_offline
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState
from tests.unit.application.post_settlement_helpers import (
    patch_settlement_poll_clear_after,
)


def test_is_transient_broker_error():
    assert is_transient_broker_error(TimeoutError("x"))
    assert is_transient_broker_error(ConnectionError("x"))
    assert is_transient_broker_error(OSError("x"))
    assert not is_transient_broker_error(ValueError("x"))


def test_is_transient_broker_error_connection_closed_name():
    assert is_transient_broker_error(type("ConnectionClosed", (Exception,), {})("x"))


def test_mark_ws_offline():
    ws = MagicMock()
    ws.is_running = True
    mark_ws_offline(ws)
    assert ws.is_running is False
    mark_ws_offline(None)


@pytest.mark.asyncio
async def test_reconcile_contracts_returns_false_when_ws_offline(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = False
        assert await reconcile_contracts(orch.executor) is False


@pytest.mark.asyncio
async def test_reconcile_contracts_logs_non_transient_error(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = True
        await orch.state.add_contract(
            Contract(
                contract_id=808,
                proposal_id="p808",
                status=TradeStatus.OPEN,
                buy_price=2.0,
                payout=4.0,
                symbol="R_75",
                direction=TradeDirection.CALL,
                stake=2.0,
                expiry_time=1,
            )
        )
        with (
            patch(
                "src.application.services.orchestrator.execution_settlement.reconcile_single_contract",
                AsyncMock(side_effect=ValueError("bad payload")),
            ),
            patch("src.application.services.orchestrator.execution_settlement.logging.getLogger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            ok = await reconcile_contracts(orch.executor)
        assert ok is True
        assert any("falhou" in str(c) for c in mock_logger.warning.call_args_list)


@pytest.mark.asyncio
async def test_reconcile_contracts_marks_ws_offline_on_timeout(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = True
        await orch.state.add_contract(
            Contract(
                contract_id=909,
                proposal_id="p909",
                status=TradeStatus.OPEN,
                buy_price=2.0,
                payout=4.0,
                symbol="R_75",
                direction=TradeDirection.CALL,
                stake=2.0,
                expiry_time=1,
            )
        )
        with patch(
            "src.application.services.orchestrator.execution_settlement.reconcile_single_contract",
            AsyncMock(side_effect=TimeoutError("timeout")),
        ):
            ok = await orch.executor.reconcile()
        assert ok is False
        assert orch.ws.is_running is False


@pytest.mark.asyncio
async def test_wait_for_settlement_preserves_pending_when_broker_offline(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = False
        orch.risk_manager.active_contract_ids = [707]
        orch.risk_manager.contract_to_symbol[707] = "R_75"
        await orch.state.add_contract(
            Contract(
                contract_id=707,
                proposal_id="p707",
                status=TradeStatus.OPEN,
                buy_price=18.0,
                payout=33.0,
                symbol="R_75",
                direction=TradeDirection.CALL,
                stake=18.0,
                expiry_time=1,
            )
        )
        seen_pending: list[list[int]] = []
        iteration = {"n": 0}

        async def stop_after_offline_polls(_seconds):
            seen_pending.append(list(orch.risk_manager.active_contract_ids))
            iteration["n"] += 1
            if iteration["n"] >= 3:
                orch.risk_manager.active_contract_ids = []

        with (
            patch(
                "src.application.services.orchestrator.execution_settlement._settlement_poll_delay",
                side_effect=stop_after_offline_polls,
            ),
            patch.object(orch.logger, "warning") as mock_warn,
            patch.object(orch, "_save_full_state", AsyncMock()),
        ):
            await orch.executor.wait_for_settlement(timeout=5)
        assert seen_pending[:2] == [[707], [707]]
        assert any("broker offline" in str(c) for c in mock_warn.call_args_list)


@pytest.mark.asyncio
async def test_wait_for_settlement_stagnant_pauses_when_ws_drops(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        ex = orch.config.setdefault("orchestrator", {}).setdefault("execution", {})
        ex["settlement_max_stagnant_polls"] = 2
        ex["settlement_stagnant_grace_seconds"] = 0.0
        ex["settlement_post_expiry_slack_seconds"] = 0.0
        orch.ws.is_running = True
        orch.risk_manager.active_contract_ids = [606]
        await orch.state.add_contract(
            Contract(
                contract_id=606,
                proposal_id="p606",
                status=TradeStatus.OPEN,
                buy_price=2.0,
                payout=4.0,
                symbol="R_75",
                direction=TradeDirection.CALL,
                stake=2.0,
                expiry_time=1,
            )
        )
        polls = {"n": 0}

        async def reconcile_then_offline():
            polls["n"] += 1
            if polls["n"] >= 3:
                orch.ws.is_running = False
            return True

        with (
            patch.object(orch.executor, "reconcile", side_effect=reconcile_then_offline),
            patch.object(orch, "_save_full_state", AsyncMock()),
            patch_settlement_poll_clear_after(orch.risk_manager, 10),
        ):
            await orch.executor.wait_for_settlement(timeout=5)
        assert polls["n"] >= 3
