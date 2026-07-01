from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.handlers.stream_reconnect import execute_stream_reconnect


@pytest.mark.asyncio
async def test_execute_stream_reconnect_success():
    orch = MagicMock()
    orch.ws.ws = MagicMock()
    orch.ws.close = AsyncMock()
    orch.auth.open_trading_session = AsyncMock(return_value=MagicMock(ws_url="wss://test", balance=100.0))
    orch.ws.connect = AsyncMock()
    orch._on_contract_update = MagicMock()
    orch._on_transaction = MagicMock()

    stream = MagicMock()
    stream.candle_callback = AsyncMock()
    stream.symbols = ["R_10"]
    stream.granularity = 300
    stream.ws = orch.ws
    stream.ws.send = AsyncMock()
    stream._on_candle = MagicMock()
    stream._on_tick = MagicMock()
    stream.tick_buffer.touch_activity = MagicMock()

    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect.ws_connect_options",
            return_value={"max_attempts": 1},
        ),
        patch(
            "src.infrastructure.handlers.stream_reconnect.subscribe_account_transactions",
            new_callable=AsyncMock,
        ),
    ):
        ok = await execute_stream_reconnect(orch, stream)

    assert ok is True
    orch.ws.close.assert_awaited_once()
    stream.tick_buffer.touch_activity.assert_called_once()


@pytest.mark.asyncio
async def test_execute_stream_reconnect_without_callback():
    orch = MagicMock()
    stream = MagicMock()
    stream.candle_callback = None
    assert await execute_stream_reconnect(orch, stream) is False


@pytest.mark.asyncio
async def test_execute_stream_reconnect_failure():
    orch = MagicMock()
    orch.ws.ws = MagicMock()
    orch.ws.close = AsyncMock()
    orch.auth.open_trading_session = AsyncMock(side_effect=RuntimeError("ws fail"))
    stream = MagicMock()
    stream.candle_callback = AsyncMock()
    with patch(
        "src.infrastructure.handlers.stream_reconnect.ws_connect_options",
        return_value={"max_attempts": 1},
    ):
        assert await execute_stream_reconnect(orch, stream) is False
