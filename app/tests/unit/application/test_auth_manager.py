from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.auth_manager import AuthManager
from src.infrastructure.api.deriv_rest_client import DerivRestError, DerivTradingSession


def test_auth_manager_reads_pat(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_test|app-1")
    auth = AuthManager(mode="demo")
    assert auth.get_pat() == "pat_test"


def test_auth_manager_pat_missing(monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_PAT", raising=False)
    with patch("src.application.services.auth_manager.load_dotenv"):
        auth = AuthManager(mode="demo")
        with pytest.raises(DerivRestError, match="AETHER_DERIV_PAT"):
            auth.rest_client()


def test_auth_manager_rest_client_requires_app_id(monkeypatch, tmp_path):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_only")
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    with (
        patch("src.application.services.auth_manager.load_dotenv"),
        patch("src.application.services.auth_manager.REPO_ROOT", tmp_path),
    ):
        auth = AuthManager(mode="demo", config={"api_config": {}})
        with pytest.raises(DerivRestError):
            auth.rest_client()


def test_auth_manager_reuses_cached_app_id(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_x|app-99")
    auth = AuthManager(mode="demo", config={"api_config": {"deriv_app_id": "app-99"}})
    auth.deriv_app_id = "cached-app"
    client = auth.rest_client()
    assert client.deriv_app_id == "cached-app"


def test_auth_manager_rejects_legacy_app_id(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_x|1089")
    auth = AuthManager(mode="demo", config={"api_config": {"deriv_app_id": "1089"}})
    auth.deriv_app_id = "1089"
    with pytest.raises(DerivRestError, match="legado"):
        auth.rest_client()


@pytest.mark.asyncio
async def test_auth_manager_open_trading_session(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_x|app-99")
    auth = AuthManager(mode="demo", config={"api_config": {"deriv_app_id": "app-99"}})
    session = DerivTradingSession(ws_url="wss://x", balance=10.0, account_id="DOT")
    with patch(
        "src.infrastructure.api.deriv_rest_client.DerivRestClient.open_trading_session",
        new_callable=AsyncMock,
        return_value=session,
    ):
        out = await auth.open_trading_session()
    assert out.account_id == "DOT"


@pytest.mark.asyncio
async def test_auth_manager_refresh_otp_ws_url(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_x|app-99")
    auth = AuthManager(mode="demo", config={"api_config": {"deriv_app_id": "app-99"}})
    with (
        patch(
            "src.application.services.auth_manager.DerivRestClient.list_accounts",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.application.services.auth_manager.select_account",
            return_value=type("A", (), {"account_id": "DOT"})(),
        ),
        patch(
            "src.application.services.auth_manager.DerivRestClient.post_otp",
            new_callable=AsyncMock,
            return_value="wss://fresh?otp=new",
        ),
    ):
        url = await auth.refresh_otp_ws_url()
    assert url == "wss://fresh?otp=new"


def test_auth_manager_loads_dotenv_if_file_exists():
    with (
        patch("pathlib.Path.is_file", return_value=True),
        patch("src.application.services.auth_manager.load_dotenv") as mock_load,
    ):
        AuthManager(mode="live")
        mock_load.assert_called()
