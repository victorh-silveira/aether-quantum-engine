from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_fractional_lots import dispatch_fractional_orders
from src.application.services.orchestrator.execution_split_abort import (
    handle_split_abort,
    next_split_attempt_seq,
)
from src.domain.models.trade import TradeDirection


def test_next_split_attempt_seq_increments_and_tracks_last():
    orch = SimpleNamespace()
    assert next_split_attempt_seq(orch) == 1
    assert orch._split_attempt_seq == 1
    assert orch._last_split_attempt_seq == 1
    assert next_split_attempt_seq(orch) == 2
    assert orch._last_split_attempt_seq == 2


@pytest.mark.asyncio
async def test_handle_split_abort_sets_state_and_logs_seq():
    orch = SimpleNamespace()
    orch.get_data_state_signature = MagicMock(return_value="sig-a")
    orch.is_trading = True
    orch._last_split_attempt_seq = 7
    logger = MagicMock()
    with patch(
        "src.application.services.orchestrator.execution_split_abort.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep_mock:
        await handle_split_abort(
            orch,
            logger,
            symbol="RDBULL",
            direction=TradeDirection.CALL,
            cycle_id=57,
        )
    assert orch.is_trading is False
    assert orch._last_split_abort_signature == "sig-a"
    assert orch._last_split_abort_symbol == "RDBULL"
    assert orch._last_split_abort_direction == "CALL"
    assert orch._last_split_abort_cycle_id == 57
    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[4] == 7
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_split_abort_without_signature_provider():
    orch = SimpleNamespace()
    orch.is_trading = True
    logger = MagicMock()
    with patch(
        "src.application.services.orchestrator.execution_split_abort.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        await handle_split_abort(
            orch,
            logger,
            symbol="RDBEAR",
            direction=TradeDirection.PUT,
            cycle_id=3,
        )
    assert orch._last_split_abort_signature == ""
    assert logger.warning.call_args.args[4] == 0


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_increments_split_attempt_seq_per_batch(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 57
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41, "date_expiry": 1710000123, "payout": 260.0}},
                {"proposal": {"id": "p-2", "ask_price": 134.41, "date_expiry": 1710000124, "payout": 260.0}},
                {"buy": {"contract_id": 1001, "buy_price": 134.41, "payout": 260.0}},
                {"buy": {"contract_id": 1002, "buy_price": 134.41, "payout": 260.0}},
                {"proposal": {"id": "p-3", "ask_price": 134.41, "date_expiry": 1710000125, "payout": 260.0}},
                {"proposal": {"id": "p-4", "ask_price": 134.41, "date_expiry": 1710000126, "payout": 260.0}},
                {"buy": {"contract_id": 1003, "buy_price": 134.41, "payout": 260.0}},
                {"buy": {"contract_id": 1004, "buy_price": 134.41, "payout": 260.0}},
            ]
        )
        orch.executor._place_order = AsyncMock()
        with patch(
            "src.application.services.orchestrator.execution_fractional_lots.subscribe_open_contract",
            new_callable=AsyncMock,
        ):
            await dispatch_fractional_orders(
                orch.executor,
                "RDBULL",
                TradeDirection.CALL,
                268.82,
                duration=60,
                metrics={"duration": 60},
                order_n=1,
            )
            await dispatch_fractional_orders(
                orch.executor,
                "RDBULL",
                TradeDirection.CALL,
                268.82,
                duration=60,
                metrics={"duration": 60},
                order_n=2,
            )
        assert orch._split_attempt_seq == 2
        second_batch_proposals = [
            call.args[0]["passthrough"]["split_attempt_seq"]
            for call in orch.ws.send.await_args_list
            if isinstance(call.args[0], dict) and "proposal" in call.args[0]
        ][2:4]
        assert second_batch_proposals == [2, 2]
