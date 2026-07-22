"""Conexao WSS com failover entre IPs resolvidos (Cloudflare anycast)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import socket
import ssl
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NoReturn
from urllib.parse import urlparse

import websockets


logger = logging.getLogger("AETH")

_state: dict[str, str | None] = {"last_good_ip": None}


def _unique_ipv4_targets(host: str, port: int) -> list[tuple[str, int]]:
    """Resolve A records IPv4 unicos preservando ordem do DNS."""
    infos = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for info in infos:
        ip = str(info[4][0])
        if ip in seen:
            continue
        seen.add(ip)
        out.append((ip, int(info[4][1])))
    return out


def _ordered_targets(host: str, port: int, *, force_ip: str | None = None) -> list[tuple[str, int]]:
    """Ordena IPs com preferencia pelo ultimo sucesso e shuffle do restante."""
    targets = _unique_ipv4_targets(host, port)
    if force_ip:
        forced = [(ip, p) for ip, p in targets if ip == force_ip]
        return forced or [(force_ip, port)]
    if not targets:
        return []
    preferred = _state.get("last_good_ip")
    rest = [t for t in targets if t[0] != preferred]
    random.shuffle(rest)
    if preferred and any(t[0] == preferred for t in targets):
        pref = next(t for t in targets if t[0] == preferred)
        return [pref] + rest
    random.shuffle(targets)
    return targets


def _tcp_connect_ip(ip: str, port: int, timeout: float) -> socket.socket:
    """Abre TCP ate um IP especifico."""
    return socket.create_connection((ip, port), timeout=max(1.0, float(timeout)))


async def _connect_one_ip(
    uri: str,
    *,
    host: str,
    ip: str,
    port: int,
    open_timeout: float,
    close_timeout: float,
    connect_kwargs: dict[str, Any],
) -> Any:
    """Tenta handshake WSS em um unico IP com SNI do host canonico."""
    sock = await asyncio.to_thread(_tcp_connect_ip, ip, port, open_timeout)
    sock.settimeout(0.0)
    headers = dict(connect_kwargs.pop("additional_headers", {}) or {})
    headers.setdefault("Origin", "https://app.deriv.com")
    try:
        return await websockets.connect(
            uri,
            sock=sock,
            ssl=ssl.create_default_context(),
            server_hostname=host,
            open_timeout=float(open_timeout),
            close_timeout=float(close_timeout),
            additional_headers=headers,
            **connect_kwargs,
        )
    except Exception:
        with contextlib.suppress(OSError):
            sock.close()
        raise


async def connect_wss_with_ip_failover(
    uri: str,
    *,
    open_timeout: float = 20.0,
    close_timeout: float = 10.0,
    per_ip_timeout: float | None = None,
    force_ip: str | None = None,
    uri_factory: Callable[[], Awaitable[str]] | None = None,
    **connect_kwargs: Any,
) -> Any:
    """Conecta WSS tentando cada IPv4; renova OTP via uri_factory entre IPs."""
    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        raise ConnectionError("WSS: URI sem host")
    scheme = (parsed.scheme or "wss").lower()
    port = int(parsed.port or (443 if scheme == "wss" else 80))
    targets = await asyncio.to_thread(_ordered_targets, host, port, force_ip=force_ip)
    if not targets or scheme != "wss":
        return await websockets.connect(
            uri,
            open_timeout=float(open_timeout),
            close_timeout=float(close_timeout),
            **connect_kwargs,
        )
    budget = max(4.0, float(open_timeout))
    ip_timeout = (
        float(per_ip_timeout) if per_ip_timeout is not None else max(3.0, min(6.0, budget / max(1, len(targets))))
    )
    last_err: BaseException | None = None
    current_uri = uri
    for index, (ip, ip_port) in enumerate(targets):
        if index > 0 and uri_factory is not None:
            current_uri = await uri_factory()
        try:
            ws = await _connect_one_ip(
                current_uri,
                host=host,
                ip=ip,
                port=ip_port,
                open_timeout=ip_timeout,
                close_timeout=close_timeout,
                connect_kwargs=dict(connect_kwargs),
            )
            _state["last_good_ip"] = ip
            if len(targets) > 1:
                logger.info("WSS: handshake OK via %s (host=%s)", ip, host)
            return ws
        except websockets.InvalidStatus as exc:
            response = getattr(exc, "response", None)
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status is None and response is not None:
                status = getattr(response, "status_code", None) or getattr(response, "status", None)
            last_err = exc
            logger.warning("WSS: IP %s status=%s (%s)", ip, status, type(exc).__name__)
            if status == 401 and uri_factory is None:
                raise
        except (
            TimeoutError,
            ConnectionError,
            OSError,
            ssl.SSLError,
            websockets.WebSocketException,
            ValueError,
            TypeError,
        ) as exc:
            last_err = exc
            logger.warning("WSS: IP %s falhou (%s): %s", ip, type(exc).__name__, exc)
    return _raise_connect_failure(last_err)


def _raise_connect_failure(last_err: BaseException | None) -> NoReturn:
    """Propaga o ultimo erro de IP ou falha generica sem tentativas."""
    if last_err is not None:
        raise last_err
    raise ConnectionError("WSS: nenhum IP resolvido para o host")


def resolved_ipv4_hosts(uri: str) -> Sequence[str]:
    """Lista IPv4 resolvidos para diagnostico."""
    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        return ()
    port = int(parsed.port or 443)
    return tuple(ip for ip, _ in _unique_ipv4_targets(host, port))
