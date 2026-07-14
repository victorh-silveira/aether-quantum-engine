from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.ws_bootstrap import setup_trading_session
from src.infrastructure.api.deriv_rest_client import DerivTradingSession


@pytest.mark.asyncio
async def test_setup_trading_session_reset_demo_balance_success(orch_config):
    orch = Orchestrator(orch_config)
    orch.auth.mode = "demo"
    session_zero = DerivTradingSession(
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo?otp=x",
        balance=0.0,
        account_id="DOT1",
    )
    session_reset = DerivTradingSession(
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo?otp=x",
        balance=10000.0,
        account_id="DOT1",
    )

    mock_client = MagicMock()
    mock_client._request = MagicMock(return_value={"data": {"balance": "10000.00"}})

    with (
        patch.object(orch.auth, "open_trading_session", AsyncMock(side_effect=[session_zero, session_reset])),
        patch.object(orch.auth, "rest_client", MagicMock(return_value=mock_client)),
    ):
        orch.ws.connect = AsyncMock()
        orch.ws.send = AsyncMock()
        orch.ws.subscribe = MagicMock()
        assert await setup_trading_session(orch) is True
        assert orch.state.balance == 10000.0


@pytest.mark.asyncio
async def test_setup_trading_session_reset_demo_balance_failure(orch_config):
    orch = Orchestrator(orch_config)
    orch.auth.mode = "demo"
    session_zero = DerivTradingSession(
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo?otp=x",
        balance=0.0,
        account_id="DOT1",
    )

    mock_client = MagicMock()
    mock_client._request = MagicMock(side_effect=RuntimeError("API Error"))

    with (
        patch.object(orch.auth, "open_trading_session", AsyncMock(return_value=session_zero)),
        patch.object(orch.auth, "rest_client", MagicMock(return_value=mock_client)),
    ):
        orch.ws.connect = AsyncMock()
        orch.ws.send = AsyncMock()
        orch.ws.subscribe = MagicMock()
        assert await setup_trading_session(orch) is True
        assert orch.state.balance == 0.0
