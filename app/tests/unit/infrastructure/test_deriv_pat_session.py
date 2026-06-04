from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.api.deriv_pat_binding import DerivPatBindingError
from src.infrastructure.api.deriv_pat_session import DerivPatSession
from src.infrastructure.api.deriv_rest_client import DerivAccount, DerivRestError


def test_deriv_pat_session_client_factory():
    session = DerivPatSession("pat_live", app_id="app-42")
    client = session._client("app-42")
    assert client.deriv_app_id == "app-42"
    assert client.access_token == "pat_live"


@pytest.mark.asyncio
async def test_deriv_pat_session_bootstrap_success(monkeypatch, tmp_path):
    monkeypatch.setattr("src.infrastructure.api.deriv_pat_session.REPO_ROOT", tmp_path)
    session = DerivPatSession("pat_live|app-1", app_id="app-1")
    session.health_check = MagicMock(return_value='{"ok":true}')
    account = DerivAccount("DOT1", 100.0, "demo", "open", "USD")
    mock_client = MagicMock()
    mock_client.list_accounts = AsyncMock(return_value=[account])
    mock_client.request_otp_ws_url = AsyncMock(return_value="wss://ws/demo")
    session._client = MagicMock(return_value=mock_client)
    result = await session.bootstrap(persist_binding=True)
    assert result.app_id == "app-1"
    assert result.ws_url == "wss://ws/demo"
    assert result.account_id == "DOT1"


@pytest.mark.asyncio
async def test_deriv_pat_session_bootstrap_binding_error(monkeypatch, tmp_path):
    monkeypatch.setattr("src.infrastructure.api.deriv_pat_session.REPO_ROOT", tmp_path)
    session = DerivPatSession("pat_x")
    session.health_check = MagicMock(return_value="ok")

    def fail(*_a, **_k):
        raise DerivPatBindingError("no app")

    session.resolve_app_id = fail
    with pytest.raises(DerivRestError):
        await session.bootstrap()


@pytest.mark.asyncio
async def test_deriv_pat_session_bootstrap_accounts_error(monkeypatch, tmp_path):
    monkeypatch.setattr("src.infrastructure.api.deriv_pat_session.REPO_ROOT", tmp_path)
    session = DerivPatSession("pat_x", app_id="app-1")
    session.health_check = MagicMock(return_value="ok")
    mock_client = MagicMock()
    mock_client.list_accounts = AsyncMock(side_effect=DerivRestError("accounts fail"))
    session._client = MagicMock(return_value=mock_client)
    with pytest.raises(DerivRestError):
        await session.bootstrap()


@pytest.mark.asyncio
async def test_deriv_pat_session_verify_websocket():
    session = DerivPatSession("pat_x", app_id="app-1")
    mock_ws = MagicMock()
    mock_ws.connect = AsyncMock()
    mock_ws.send = AsyncMock(return_value={"time": 1})
    mock_ws.close = AsyncMock()
    with patch("src.infrastructure.api.deriv_pat_session.WebSocketManager", return_value=mock_ws):
        payload = await session.verify_websocket("wss://test")
    assert payload["time"] == 1
    mock_ws.close.assert_awaited_once()


def test_deriv_pat_session_health_and_resolve(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.api.deriv_pat_session.read_http_response",
        lambda *_a, **_k: b'{"status":"ok"}',
    )
    session = DerivPatSession("pat_x|app-9")
    assert "ok" in session.health_check()
    with patch(
        "src.infrastructure.api.deriv_pat_session.discover_app_id_for_pat",
        return_value="app-9",
    ):
        assert session.resolve_app_id() == "app-9"
