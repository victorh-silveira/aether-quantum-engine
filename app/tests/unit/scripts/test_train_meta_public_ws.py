from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.operations.train_meta_data import _open_deriv_ws, _public_ws_url


def test_public_ws_url_from_settings():
    assert (
        _public_ws_url({"api_config": {"public_ws_url": "wss://api.derivws.com/trading/v1/options/ws/public"}})
        == "wss://api.derivws.com/trading/v1/options/ws/public"
    )


def test_public_ws_url_missing_raises():
    with pytest.raises(RuntimeError, match="public_ws_url"):
        _public_ws_url({})


@pytest.mark.asyncio
async def test_open_deriv_ws_uses_public_gateway():
    mgr = MagicMock()
    mgr.uri = "wss://api.derivws.com/trading/v1/options/ws/public"
    mgr.connect = AsyncMock()
    with patch("scripts.operations.train_meta_data.WebSocketManager", return_value=mgr) as ctor:
        out = await _open_deriv_ws(
            {
                "api_config": {
                    "public_ws_url": "wss://api.derivws.com/trading/v1/options/ws/public",
                    "request_timeout_seconds": 45,
                }
            }
        )
    assert out is mgr
    ctor.assert_called_once_with(
        "wss://api.derivws.com/trading/v1/options/ws/public",
        request_timeout=45,
    )
    mgr.connect.assert_awaited_once()
