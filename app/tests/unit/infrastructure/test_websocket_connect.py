from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from src.infrastructure.api import websocket_connect as wsc
from src.infrastructure.api.websocket_connect import (
    _unique_ipv4_targets,
    connect_wss_with_ip_failover,
    resolved_ipv4_hosts,
)


def test_unique_ipv4_targets_dedupes():
    infos = [
        (2, 1, 6, "", ("1.1.1.1", 443)),
        (2, 1, 6, "", ("1.1.1.1", 443)),
        (2, 1, 6, "", ("2.2.2.2", 443)),
    ]
    with patch("socket.getaddrinfo", return_value=infos):
        assert _unique_ipv4_targets("api.example", 443) == [("1.1.1.1", 443), ("2.2.2.2", 443)]
        assert resolved_ipv4_hosts("wss://api.example/path") == ("1.1.1.1", "2.2.2.2")
        assert resolved_ipv4_hosts("wss:///nohost") == ()


@pytest.mark.asyncio
async def test_connect_failover_skips_timeout_ip_then_succeeds():
    wsc._state["last_good_ip"] = None
    ws = AsyncMock()
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=[TimeoutError("stall"), ws],
        ) as mock_one,
    ):
        result = await connect_wss_with_ip_failover("wss://api.example/ws", open_timeout=16.0, per_ip_timeout=4.0)
    assert result is ws
    assert mock_one.await_count == 2
    assert wsc._state["last_good_ip"] == "2.2.2.2"


@pytest.mark.asyncio
async def test_connect_failover_refreshes_uri_between_ips():
    wsc._state["last_good_ip"] = None
    ws = AsyncMock()
    factory = AsyncMock(return_value="wss://api.example/ws?otp=new")
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=[TimeoutError("stall"), ws],
        ),
    ):
        result = await connect_wss_with_ip_failover(
            "wss://api.example/ws?otp=old",
            open_timeout=12.0,
            per_ip_timeout=3.0,
            uri_factory=factory,
        )
    assert result is ws
    factory.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_failover_raises_401_without_factory():
    response = MagicMock()
    response.status_code = 401
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=websockets.InvalidStatus(response),
        ),
        pytest.raises(websockets.InvalidStatus),
    ):
        await connect_wss_with_ip_failover("wss://api.example/ws", open_timeout=10.0, per_ip_timeout=3.0)


@pytest.mark.asyncio
async def test_connect_without_targets_uses_direct_connect():
    ws = AsyncMock()
    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[]),
        patch("websockets.connect", new_callable=AsyncMock, return_value=ws) as mock_ws,
    ):
        result = await connect_wss_with_ip_failover("wss://api.example/ws", open_timeout=5.0)
    assert result is ws
    assert mock_ws.await_count == 1


@pytest.mark.asyncio
async def test_connect_one_ip_closes_sock_on_failure():
    sock = MagicMock()
    with (
        patch("src.infrastructure.api.websocket_connect._tcp_connect_ip", return_value=sock),
        patch("websockets.connect", new_callable=AsyncMock, side_effect=TimeoutError("x")),
        pytest.raises(TimeoutError),
    ):
        await wsc._connect_one_ip(
            "wss://api.example/ws",
            host="api.example",
            ip="1.1.1.1",
            port=443,
            open_timeout=3.0,
            close_timeout=1.0,
            connect_kwargs={},
        )
    sock.settimeout.assert_called_with(0.0)
    sock.close.assert_called()


def test_ordered_targets_force_and_preferred():
    wsc._state["last_good_ip"] = "2.2.2.2"
    with patch(
        "src.infrastructure.api.websocket_connect._unique_ipv4_targets",
        return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
    ):
        forced = wsc._ordered_targets("api.example", 443, force_ip="1.1.1.1")
        assert forced == [("1.1.1.1", 443)]
        ordered = wsc._ordered_targets("api.example", 443)
        assert ordered[0] == ("2.2.2.2", 443)
    wsc._state["last_good_ip"] = None
    with patch("src.infrastructure.api.websocket_connect._unique_ipv4_targets", return_value=[]):
        assert wsc._ordered_targets("api.example", 443) == []
    with patch(
        "src.infrastructure.api.websocket_connect._unique_ipv4_targets",
        return_value=[("9.9.9.9", 443)],
    ):
        assert wsc._ordered_targets("api.example", 443) == [("9.9.9.9", 443)]
    with patch("src.infrastructure.api.websocket_connect._unique_ipv4_targets", return_value=[]):
        assert wsc._ordered_targets("api.example", 443, force_ip="8.8.8.8") == [("8.8.8.8", 443)]


def test_tcp_connect_ip_calls_create_connection():
    with patch("socket.create_connection", return_value=MagicMock()) as mock_conn:
        wsc._tcp_connect_ip("1.2.3.4", 443, 2.5)
    mock_conn.assert_called_once_with(("1.2.3.4", 443), timeout=2.5)


@pytest.mark.asyncio
async def test_connect_raises_without_host():
    with pytest.raises(ConnectionError, match="URI sem host"):
        await connect_wss_with_ip_failover("wss:///path", open_timeout=2.0)


@pytest.mark.asyncio
async def test_connect_raises_when_all_ips_fail():
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=TimeoutError("stall"),
        ),
        pytest.raises(TimeoutError),
    ):
        await connect_wss_with_ip_failover("wss://api.example/ws", open_timeout=6.0, per_ip_timeout=2.0)


@pytest.mark.asyncio
async def test_connect_failover_otp_without_factory_aborts_after_first_failure():
    wsc._state["last_good_ip"] = None
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=TimeoutError("stall"),
        ) as mock_one,
        pytest.raises(ConnectionError, match="OTP"),
    ):
        await connect_wss_with_ip_failover(
            "wss://api.example/ws?otp=burned",
            open_timeout=12.0,
            per_ip_timeout=3.0,
        )
    assert mock_one.await_count == 1


@pytest.mark.asyncio
async def test_connect_failover_otp_401_refreshes_with_factory():
    wsc._state["last_good_ip"] = None
    response = MagicMock()
    response.status_code = 401
    ws = AsyncMock()
    factory = AsyncMock(return_value="wss://api.example/ws?otp=new")
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=[websockets.InvalidStatus(response), ws],
        ),
    ):
        result = await connect_wss_with_ip_failover(
            "wss://api.example/ws?otp=old",
            open_timeout=12.0,
            per_ip_timeout=3.0,
            uri_factory=factory,
        )
    assert result is ws
    factory.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_failover_otp_non_401_without_factory_aborts():
    wsc._state["last_good_ip"] = None
    response = MagicMock()
    response.status_code = 503
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new_callable=AsyncMock,
            side_effect=websockets.InvalidStatus(response),
        ) as mock_one,
        pytest.raises(ConnectionError, match="OTP one-shot"),
    ):
        await connect_wss_with_ip_failover(
            "wss://api.example/ws?otp=x",
            open_timeout=12.0,
            per_ip_timeout=3.0,
        )
    assert mock_one.await_count == 1


def test_uri_has_otp_and_status_helpers():
    assert wsc._uri_has_otp("wss://api.example/ws?otp=abc") is True
    assert wsc._uri_has_otp("wss://api.example/trading/v1/options/ws/public") is False
    response = MagicMock()
    response.status_code = 503
    exc = websockets.InvalidStatus(response)
    assert wsc._status_code(exc) == 503
    bare = MagicMock(spec=[])
    bare.status_code = 502
    bare.status = None
    bare.response = None
    assert wsc._status_code(bare) == 502
    empty = MagicMock(spec=[])
    empty.status_code = None
    empty.status = None
    empty.response = None
    assert wsc._status_code(empty) is None


def test_raise_connect_failure_branches():
    with pytest.raises(TimeoutError):
        wsc._raise_connect_failure(TimeoutError("x"))
    with pytest.raises(ConnectionError, match="nenhum IP"):
        wsc._raise_connect_failure(None)
