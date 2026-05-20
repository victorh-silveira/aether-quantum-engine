import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
import websockets

from src.infrastructure.api.websocket_manager import WebSocketManager


@pytest.fixture
def ws_manager():
    return WebSocketManager("ws://test", request_timeout=1, ping_interval=0.1)


@pytest.mark.asyncio
async def test_ws_connect_and_close(ws_manager):
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = AsyncMock()
        await ws_manager.connect()
        assert ws_manager.is_running is True
        await ws_manager.close()
        assert ws_manager.is_running is False


@pytest.mark.asyncio
async def test_ws_send_receive_flow(ws_manager):
    ws_manager.ws = AsyncMock()
    ws_manager.is_running = True

    future = asyncio.get_event_loop().create_future()
    ws_manager.callbacks[1] = future
    future.set_result({"req_id": 1, "msg": "ok"})

    with patch("asyncio.wait_for", return_value={"msg": "ok"}):
        res = await ws_manager.send({"cmd": "test"})
        assert res["msg"] == "ok"


@pytest.mark.asyncio
async def test_ws_send_errors(ws_manager):
    ws_manager.ws = None
    before = ws_manager.req_id_counter
    with pytest.raises(ConnectionError):
        await ws_manager.send({"cmd": "test"})
    assert ws_manager.req_id_counter == before
    assert ws_manager.callbacks == {}

    ws_manager.ws = AsyncMock()
    ws_manager.is_running = True
    with pytest.raises(asyncio.TimeoutError):
        await ws_manager.send({"cmd": "timeout"}, timeout=0.01)


@pytest.mark.asyncio
async def test_ws_listen_logic(ws_manager):
    ws_manager.ws = AsyncMock()
    data = json.dumps({"req_id": 1, "msg_type": "tick", "tick": {"q": 1}})
    ws_manager.ws.__aiter__.return_value = [data]

    mock_cb = AsyncMock()
    ws_manager.subscribe("tick", mock_cb)

    future = asyncio.get_event_loop().create_future()
    ws_manager.callbacks[1] = future

    task = asyncio.create_task(ws_manager._listen())
    await asyncio.sleep(0.05)
    ws_manager.is_running = False
    await task

    assert mock_cb.called
    assert future.done()
    assert (await future)["req_id"] == 1


@pytest.mark.asyncio
async def test_ws_listen_exceptions(ws_manager):
    ws_manager.ws = AsyncMock()
    ws_manager.ws.__aiter__.side_effect = websockets.ConnectionClosed(None, None)
    ws_manager.is_running = True
    await ws_manager._listen()
    assert ws_manager.is_running is False

    ws_manager.ws.__aiter__.side_effect = Exception("error")
    ws_manager.is_running = True
    await ws_manager._listen()
    assert ws_manager.is_running is False


@pytest.mark.asyncio
async def test_ws_ping_loop(ws_manager):
    ws_manager.ws = AsyncMock()
    ws_manager.is_running = True

    async def stop_after_delay():
        await asyncio.sleep(0.15)
        ws_manager.is_running = False

    await asyncio.gather(ws_manager._ping_loop(), stop_after_delay())
    assert ws_manager.ws.send.called


@pytest.mark.asyncio
async def test_ws_ping_loop_exception(ws_manager):
    ws_manager.ws = AsyncMock()
    ws_manager.ws.send.side_effect = Exception("ping_fail")
    ws_manager.is_running = True
    await ws_manager._ping_loop()
    assert ws_manager.ws.send.called
    assert ws_manager.is_running is False
    assert ws_manager.ws.close.called


@pytest.mark.asyncio
async def test_ws_ping_loop_critical_fail(ws_manager):
    ws_manager.ws = AsyncMock()
    ws_manager.is_running = True

    with (
        patch("asyncio.sleep", side_effect=Exception("critical_fail")),
        patch.object(ws_manager.logger, "error") as mock_log,
    ):
        await ws_manager._ping_loop()
        assert mock_log.called
        assert ws_manager.is_running is False


@pytest.mark.asyncio
async def test_ws_heuristic_routing(ws_manager):
    ws_manager.ws = AsyncMock()
    data_ohlc = json.dumps({"ohlc": {"symbol": "frxEURUSD"}})
    data_poc = json.dumps({"proposal_open_contract": {"id": 1}})

    ws_manager.ws.__aiter__.return_value = [data_ohlc, data_poc]

    cb_ohlc = AsyncMock()
    cb_poc = AsyncMock()
    ws_manager.subscribe("ohlc", cb_ohlc)
    ws_manager.subscribe("proposal_open_contract", cb_poc)

    task = asyncio.create_task(ws_manager._listen())
    await asyncio.sleep(0.1)
    ws_manager.is_running = False
    await task

    assert cb_ohlc.called
    assert cb_poc.called


@pytest.mark.asyncio
async def test_ws_callback_error_handling(ws_manager):
    ws_manager.ws = AsyncMock()
    data = json.dumps({"msg_type": "fail", "data": "test"})
    ws_manager.ws.__aiter__.return_value = [data]

    async def failing_cb(d):
        raise Exception("Callback Fail")

    ws_manager.subscribe("fail", failing_cb)

    with patch.object(ws_manager.logger, "error") as mock_log:
        task = asyncio.create_task(ws_manager._listen())
        await asyncio.sleep(0.1)
        ws_manager.is_running = False
        await task
        assert mock_log.called
