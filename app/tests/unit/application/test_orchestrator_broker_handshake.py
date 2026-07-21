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
