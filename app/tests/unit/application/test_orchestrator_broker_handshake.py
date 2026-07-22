import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.ws_bootstrap import open_broker_handshake, setup_trading_session


@pytest.mark.asyncio
async def test_open_broker_handshake_timeout_raises_runtime_error(orch_config):
    orch = Orchestrator(orch_config)
    stuck = asyncio.Event()

    async def _hang(*_args: object, **_kwargs: object) -> object:
        await stuck.wait()
        raise AssertionError("unreachable")

    orch.auth.open_trading_session = _hang
    with (
        patch(
            "src.application.services.orchestrator.ws_bootstrap.resolve_orchestrator_timing_config",
            return_value={"broker_handshake_timeout_seconds": 0.05, "ws_connect": {}},
        ),
        pytest.raises(RuntimeError, match="HANDSHAKE_TIMEOUT"),
    ):
        await open_broker_handshake(orch)


@pytest.mark.asyncio
async def test_setup_trading_session_plain_timeout_error(orch_config):
    orch = Orchestrator(orch_config)
    with (
        patch(
            "src.application.services.orchestrator.ws_bootstrap.validate_infra_services",
            AsyncMock(side_effect=TimeoutError("slow")),
        ),
        patch.object(orch.logger, "warning") as mock_warn,
    ):
        assert await setup_trading_session(orch) is False
    assert any("broker indisponivel" in str(c) for c in mock_warn.call_args_list)


@pytest.mark.asyncio
async def test_broker_handshake_refreshes_otp_via_uri_factory(orch_config):
    from src.application.services.orchestrator.ws_bootstrap import _broker_pat_websocket_handshake
    from src.infrastructure.api.deriv_rest_client import DerivTradingSession

    orch = Orchestrator(orch_config)
    first = DerivTradingSession(ws_url="wss://api.example/ws?otp=1", balance=100.0, account_id="DOT1")
    second = DerivTradingSession(ws_url="wss://api.example/ws?otp=2", balance=100.0, account_id="DOT1")
    orch.auth.open_trading_session = AsyncMock(side_effect=[first, second])
    orch.auth.mode = "demo"

    async def _connect(uri, **kwargs):
        factory = kwargs.get("uri_factory")
        assert factory is not None
        refreshed = await factory()
        assert refreshed.endswith("otp=2")

    orch.ws.connect = AsyncMock(side_effect=_connect)
    with patch(
        "src.application.services.orchestrator.ws_bootstrap.ws_connect_options",
        return_value={"max_attempts": 1, "open_timeout": 5.0, "retry_delay": 0.1, "retry_backoff": 1.0},
    ):
        session = await _broker_pat_websocket_handshake(orch)
    assert session.ws_url.endswith("otp=2")


@pytest.mark.asyncio
async def test_setup_trading_session_broker_handshake_timeout(orch_config):
    orch = Orchestrator(orch_config)
    with (
        patch(
            "src.application.services.orchestrator.ws_bootstrap.validate_infra_services",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.meta_classifier_enabled",
            return_value=False,
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.bootstrap_and_validate_models",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.restore_orchestrator_state",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.open_broker_handshake",
            AsyncMock(
                side_effect=RuntimeError(
                    "[AETHER] HANDSHAKE_TIMEOUT: WebSocket/Deriv estagnou (rede ou firewall). "
                    "TCP silent drop ou barreira local bloqueou o aperto de mao seguro."
                )
            ),
        ),
        patch.object(orch.logger, "error") as mock_error,
    ):
        assert await setup_trading_session(orch) is False
    assert any("HANDSHAKE_TIMEOUT" in str(c) for c in mock_error.call_args_list)
