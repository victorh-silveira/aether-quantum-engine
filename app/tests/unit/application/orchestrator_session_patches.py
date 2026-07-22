from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch


@contextmanager
def session_setup_patches(*, otp_ok: bool, public_side_effect=None):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
                AsyncMock(return_value=("DOT1", 1000.0)),
            )
        )
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap.bootstrap_and_validate_models",
                AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap.restore_orchestrator_state",
                AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap.bootstrap_active_session_targets",
                AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap.open_public_market_handshake",
                AsyncMock(side_effect=public_side_effect),
            )
        )
        stack.enter_context(
            patch(
                "src.application.services.orchestrator.ws_bootstrap._try_optional_otp_trading_ws",
                AsyncMock(return_value=otp_ok),
            )
        )
        yield
