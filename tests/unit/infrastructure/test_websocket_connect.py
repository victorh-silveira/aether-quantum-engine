"""Testes unitarios para failover de IPv4 e OTP em websocket_connect."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from src.infrastructure.api import websocket_connect as wc


def test_unique_ipv4_targets_and_ordered_targets():
    infos = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("2.2.2.2", 443)),
    ]
    with patch("src.infrastructure.api.websocket_connect.socket.getaddrinfo", return_value=infos):
        assert wc._unique_ipv4_targets("ws.derivws.com", 443) == [("1.1.1.1", 443), ("2.2.2.2", 443)]
        assert wc._ordered_targets("ws.derivws.com", 443, force_ip="2.2.2.2") == [("2.2.2.2", 443)]
        assert wc._ordered_targets("ws.derivws.com", 443, force_ip="9.9.9.9") == [("9.9.9.9", 443)]
        wc._state["last_good_ip"] = "2.2.2.2"
        ordered = wc._ordered_targets("ws.derivws.com", 443)
        assert ordered[0] == ("2.2.2.2", 443)
        wc._state["last_good_ip"] = None
        assert len(wc._ordered_targets("ws.derivws.com", 443)) == 2
    with patch("src.infrastructure.api.websocket_connect.socket.getaddrinfo", return_value=[]):
        assert wc._ordered_targets("ws.derivws.com", 443) == []
        assert wc.resolved_ipv4_hosts("wss://ws.derivws.com/v3") == ()
    assert wc.resolved_ipv4_hosts("not-a-uri") == ()


def test_status_code_and_tcp_connect_and_failure_helpers():
    exc1 = SimpleNamespace(status_code=401)
    assert wc._status_code(exc1) == 401
    exc2 = SimpleNamespace(response=SimpleNamespace(status=503))
    assert wc._status_code(exc2) == 503
    assert wc._status_code(SimpleNamespace()) is None

    fake_sock = MagicMock()
    with patch("src.infrastructure.api.websocket_connect.socket.create_connection", return_value=fake_sock):
        assert wc._tcp_connect_ip("1.1.1.1", 443, 5.0) is fake_sock
    fake_sock.setsockopt.assert_called_once()

    with pytest.raises(ConnectionError, match="nenhum IP"):
        wc._raise_connect_failure(None)
    with pytest.raises(RuntimeError, match="boom"):
        wc._raise_connect_failure(RuntimeError("boom"))


@pytest.mark.asyncio
async def test_connect_one_ip_success_and_cleanup():
    sock = MagicMock()
    ws_conn = MagicMock()
    with (
        patch("src.infrastructure.api.websocket_connect._tcp_connect_ip", return_value=sock),
        patch("src.infrastructure.api.websocket_connect.websockets.connect", new=AsyncMock(return_value=ws_conn)),
    ):
        res = await wc._connect_one_ip(
            "wss://ws.derivws.com/v3",
            host="ws.derivws.com",
            ip="1.1.1.1",
            port=443,
            open_timeout=5.0,
            close_timeout=5.0,
            connect_kwargs={},
        )
    assert res is ws_conn

    with (
        patch("src.infrastructure.api.websocket_connect._tcp_connect_ip", return_value=sock),
        patch(
            "src.infrastructure.api.websocket_connect.websockets.connect", new=AsyncMock(side_effect=OSError("fail"))
        ),
        pytest.raises(OSError, match="fail"),
    ):
        await wc._connect_one_ip(
            "wss://ws.derivws.com/v3",
            host="ws.derivws.com",
            ip="1.1.1.1",
            port=443,
            open_timeout=5.0,
            close_timeout=5.0,
            connect_kwargs={},
        )
    sock.close.assert_called_once()


@pytest.mark.asyncio
async def test_connect_wss_with_ip_failover_branches():
    with pytest.raises(ConnectionError, match="URI sem host"):
        await wc.connect_wss_with_ip_failover("ws://")

    ws_direct = MagicMock()
    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[]),
        patch("src.infrastructure.api.websocket_connect.websockets.connect", new=AsyncMock(return_value=ws_direct)),
    ):
        assert await wc.connect_wss_with_ip_failover("wss://ws.derivws.com/v3") is ws_direct

    # Failover com uri_factory renovando OTP apos InvalidStatus e TimeoutError
    ws_ok = MagicMock()
    inv_401 = websockets.InvalidStatus(websockets.Response(401, "Unauthorized", websockets.Headers()))
    inv_503 = websockets.InvalidStatus(websockets.Response(503, "Unavailable", websockets.Headers()))
    with (
        patch(
            "src.infrastructure.api.websocket_connect._ordered_targets",
            return_value=[("1.1.1.1", 443), ("2.2.2.2", 443), ("3.3.3.3", 443)],
        ),
        patch(
            "src.infrastructure.api.websocket_connect._connect_one_ip",
            new=AsyncMock(side_effect=[inv_503, TimeoutError("t"), ws_ok]),
        ),
    ):
        res = await wc.connect_wss_with_ip_failover(
            "wss://ws.derivws.com/v3?otp=1",
            uri_factory=AsyncMock(return_value="wss://ws.derivws.com/v3?otp=2"),
        )
    assert res is ws_ok

    # 401 sem uri_factory propaga direto
    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[("1.1.1.1", 443)]),
        patch("src.infrastructure.api.websocket_connect._connect_one_ip", new=AsyncMock(side_effect=inv_401)),
        pytest.raises(websockets.InvalidStatus),
    ):
        await wc.connect_wss_with_ip_failover("wss://ws.derivws.com/v3")

    # OTP locked sem uri_factory apos InvalidStatus non-401 e TimeoutError
    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[("1.1.1.1", 443)]),
        patch("src.infrastructure.api.websocket_connect._connect_one_ip", new=AsyncMock(side_effect=inv_503)),
        pytest.raises(ConnectionError, match="OTP one-shot"),
    ):
        await wc.connect_wss_with_ip_failover("wss://ws.derivws.com/v3?otp=1")

    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[("1.1.1.1", 443)]),
        patch("src.infrastructure.api.websocket_connect._connect_one_ip", new=AsyncMock(side_effect=TimeoutError("t"))),
        pytest.raises(ConnectionError, match="consumido o OTP"),
    ):
        await wc.connect_wss_with_ip_failover("wss://ws.derivws.com/v3?otp=1")

    # Todos os IPs falham sem OTP
    with (
        patch("src.infrastructure.api.websocket_connect._ordered_targets", return_value=[("1.1.1.1", 443)]),
        patch("src.infrastructure.api.websocket_connect._connect_one_ip", new=AsyncMock(side_effect=TimeoutError("t"))),
        pytest.raises(TimeoutError),
    ):
        await wc.connect_wss_with_ip_failover("wss://ws.derivws.com/v3")
