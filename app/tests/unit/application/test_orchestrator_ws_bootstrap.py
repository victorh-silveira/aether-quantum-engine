from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.ws_bootstrap import (
    setup_trading_session,
    start_orchestrator_streams,
    subscribe_account_transactions,
)
from src.infrastructure.api.deriv_rest_client import DerivRestError, DerivTradingSession


@pytest.mark.asyncio
async def test_setup_trading_session_success(orch_config):
    orch = Orchestrator(orch_config)
    session = DerivTradingSession(
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo?otp=x",
        balance=100.0,
        account_id="DOT1",
    )
    with patch.object(orch.auth, "open_trading_session", AsyncMock(return_value=session)):
        orch.ws.connect = AsyncMock()
        orch.ws.send = AsyncMock()
        orch.ws.subscribe = MagicMock()
        assert await setup_trading_session(orch) is True
        assert orch.state.balance == 100.0


@pytest.mark.asyncio
async def test_setup_trading_session_rest_error(orch_config):
    orch = Orchestrator(orch_config)
    with patch.object(orch.auth, "open_trading_session", AsyncMock(side_effect=DerivRestError("fail"))):
        assert await setup_trading_session(orch) is False


@pytest.mark.asyncio
async def test_subscribe_account_transactions_success(orch_config):
    orch = Orchestrator(orch_config)
    orch.ws.send = AsyncMock()
    orch.ws.subscribe = MagicMock()
    await subscribe_account_transactions(orch)
    orch.ws.send.assert_awaited_once()
    orch.ws.subscribe.assert_called_once()


@pytest.mark.asyncio
async def test_setup_trading_session_closes_existing_ws(orch_config):
    orch = Orchestrator(orch_config)
    orch.ws.ws = MagicMock()
    orch.ws.close = AsyncMock()
    session = DerivTradingSession(
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo?otp=x",
        balance=50.0,
        account_id="DOT1",
    )
    with patch.object(orch.auth, "open_trading_session", AsyncMock(return_value=session)):
        orch.ws.connect = AsyncMock()
        orch.ws.send = AsyncMock()
        orch.ws.subscribe = MagicMock()
        assert await setup_trading_session(orch) is True
        orch.ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_orchestrator_streams_success(orch_config):
    orch = Orchestrator(orch_config)
    orch.stream.start_candle_stream = AsyncMock()
    orch.ws.is_running = True
    assert await start_orchestrator_streams(orch) is True


@pytest.mark.asyncio
async def test_start_orchestrator_streams_retries_then_fails(orch_config):
    orch = Orchestrator(orch_config)
    orch.ws.is_running = False
    orch.ws.connect = AsyncMock()
    orch.stream.start_candle_stream = AsyncMock(side_effect=[ConnectionError("x"), ConnectionError("y")])
    assert await start_orchestrator_streams(orch) is False
