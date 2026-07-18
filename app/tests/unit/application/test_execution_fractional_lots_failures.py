from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_fractional_lots import dispatch_fractional_orders
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_returns_empty_when_single_lot_fails(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.executor._place_order = AsyncMock(return_value=None)
        contracts = await dispatch_fractional_orders(
            orch.executor, "RDBULL", TradeDirection.CALL, 150.0, duration=60, metrics={"duration": 60}, order_n=1
        )
        assert contracts == []


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_aborts_cluster_when_proposal_fails(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.executor._place_order = AsyncMock()
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
                {"error": {"message": "Unknown contract proposal"}},
            ]
        )
        metrics = {"duration": 60}
        with patch(
            "src.application.services.orchestrator.execution_fractional_lots_buy.subscribe_open_contract",
            new_callable=AsyncMock,
        ) as subscribe_mock:
            contracts = await dispatch_fractional_orders(
                orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics=metrics, order_n=1
            )
        assert contracts == []
        assert metrics["fractional_lot_technical_failure"] is True
        assert orch.ws.send.await_count == 2
        subscribe_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_aborts_when_proposal_id_is_reused(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.executor._place_order = AsyncMock()
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
            ]
        )
        metrics = {"duration": 60}
        contracts = await dispatch_fractional_orders(
            orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics=metrics, order_n=1
        )
        assert contracts == []
        assert metrics["fractional_lot_technical_failure"] is True


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_aborts_when_proposal_response_is_not_dict(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.send = AsyncMock(return_value="invalid")
        metrics = {"duration": 60}
        contracts = await dispatch_fractional_orders(
            orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics=metrics, order_n=1
        )
        assert contracts == []
        assert metrics["fractional_lot_technical_failure"] is True


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_aborts_when_proposal_payload_missing(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.send = AsyncMock(return_value={"ok": True})
        metrics = {"duration": 60}
        contracts = await dispatch_fractional_orders(
            orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics=metrics, order_n=1
        )
        assert contracts == []
        assert metrics["fractional_lot_technical_failure"] is True


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_aborts_when_proposal_id_missing(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.send = AsyncMock(return_value={"proposal": {"ask_price": 134.41}})
        metrics = {"duration": 60}
        contracts = await dispatch_fractional_orders(
            orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics=metrics, order_n=1
        )
        assert contracts == []
        assert metrics["fractional_lot_technical_failure"] is True


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_raises_when_buy_response_missing_payload(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.executor._place_order = AsyncMock()
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
                {"proposal": {"id": "p-2", "ask_price": 134.41}},
                {"buy": {"contract_id": 1001, "buy_price": 134.41}},
                {"unexpected": True},
            ]
        )
        with pytest.raises(RuntimeError, match="resposta sem buy"):
            await dispatch_fractional_orders(
                orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics={"duration": 60}, order_n=1
            )


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_raises_when_buy_response_not_dict(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
                {"proposal": {"id": "p-2", "ask_price": 134.41}},
                "invalid",
            ]
        )
        with pytest.raises(RuntimeError, match="resposta invalida"):
            await dispatch_fractional_orders(
                orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics={"duration": 60}, order_n=1
            )


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_raises_when_buy_response_has_error(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41}},
                {"proposal": {"id": "p-2", "ask_price": 134.41}},
                {"error": {"message": "Unknown contract proposal"}},
            ]
        )
        with pytest.raises(RuntimeError, match="Unknown contract proposal"):
            await dispatch_fractional_orders(
                orch.executor, "RDBULL", TradeDirection.CALL, 268.82, duration=60, metrics={"duration": 60}, order_n=1
            )
