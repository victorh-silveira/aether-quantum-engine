from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.ws_bootstrap import setup_trading_session
from src.infrastructure.api.deriv_rest_client import DerivAccount


@pytest.mark.asyncio
async def test_setup_execute_falls_back_to_public_and_rest(orch_config):
    orch = Orchestrator(orch_config)
    client = MagicMock()
    client.list_accounts = AsyncMock(
        return_value=[
            DerivAccount(
                account_id="DOT1",
                balance=500.0,
                account_type="demo",
                status="active",
                currency="USD",
            )
        ]
    )
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
            "src.application.services.orchestrator.ws_bootstrap.bootstrap_active_session_targets",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._try_optional_otp_trading_ws",
            AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.open_public_market_handshake",
            AsyncMock(),
        ) as mock_public,
        patch.object(orch.auth, "rest_client", return_value=client),
    ):
        assert await setup_trading_session(orch) is True
    mock_public.assert_awaited_once()
    assert orch.trading_transport == "rest"
    assert orch.trade_handler.trading_transport == "rest"
    assert orch.deriv_account_id == "DOT1"


@pytest.mark.asyncio
async def test_try_optional_otp_trading_ws_returns_false_on_failure(orch_config):
    from src.application.services.orchestrator.ws_bootstrap import _try_optional_otp_trading_ws

    orch = Orchestrator(orch_config)
    with patch(
        "src.application.services.orchestrator.ws_bootstrap._broker_pat_websocket_handshake",
        AsyncMock(side_effect=TimeoutError("otp blocked")),
    ):
        assert await _try_optional_otp_trading_ws(orch) is False


@pytest.mark.asyncio
async def test_otp_fallback_closes_stale_ws_before_public(orch_config):
    orch = Orchestrator(orch_config)
    orch.ws.ws = MagicMock()
    close_calls = {"n": 0}

    async def _close() -> None:
        close_calls["n"] += 1
        if close_calls["n"] == 1:
            orch.ws.ws = MagicMock()
        else:
            orch.ws.ws = None

    orch.ws.close = AsyncMock(side_effect=_close)
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
            "src.application.services.orchestrator.ws_bootstrap.bootstrap_active_session_targets",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
            AsyncMock(return_value=("DOT1", 500.0)),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._try_optional_otp_trading_ws",
            AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.open_public_market_handshake",
            AsyncMock(),
        ),
    ):
        assert await setup_trading_session(orch) is True
    assert close_calls["n"] >= 2
    assert orch.trading_transport == "rest"
