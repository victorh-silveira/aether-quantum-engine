"""Adapter de market data publico da Deriv, sem autenticacao de conta."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import websockets


PUBLIC_URL = "wss://api.derivws.com/trading/v1/options/ws/public"


class PublicDeriv:
    """Cliente estritamente limitado a catalogo e historico OHLC."""

    def __init__(
        self,
        url: str = PUBLIC_URL,
        request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        *,
        on_rtt: Callable[[str, float, bool], None] | None = None,
    ) -> None:
        self.url = url
        self._request_override = request
        self._on_rtt = on_rtt
        self._socket: Any = None
        self._lock = asyncio.Lock()
        self._request_id = 0

    def _emit_rtt(self, method: str, started_at: float, *, ok: bool) -> None:
        """Notifica callback opcional de latencia sem propagar erro de metrica."""
        if self._on_rtt is not None:
            self._on_rtt(method, max(0.0, (time.monotonic() - started_at) * 1000.0), ok=ok)

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Envia requisicao publica autorizada com reconexao unica."""
        if set(payload) - {"active_symbols", "ticks_history", "style", "granularity", "count", "end"}:
            raise ValueError("operacao nao permitida no indicador")
        method = "active_symbols" if "active_symbols" in payload else "ticks_history"
        started_at = time.monotonic()
        if self._request_override is not None:
            try:
                res = await self._request_override(payload)
                self._emit_rtt(method, started_at, ok=True)
                return res
            except Exception:
                self._emit_rtt(method, started_at, ok=False)
                raise
        async with self._lock:
            for attempt in range(2):
                try:
                    if self._socket is None:
                        self._socket = await websockets.connect(
                            self.url, open_timeout=15, max_size=8_000_000, ping_interval=20.0, ping_timeout=10.0
                        )
                    self._request_id += 1
                    await self._socket.send(json.dumps({**payload, "req_id": self._request_id}))
                    response = json.loads(await asyncio.wait_for(self._socket.recv(), timeout=20))
                    if response.get("error"):
                        raise RuntimeError(str(response["error"].get("message", response["error"])))
                    if response.get("req_id") not in (None, self._request_id):
                        raise RuntimeError("resposta fora de ordem")
                    self._emit_rtt(method, started_at, ok=True)
                    return response
                except (TimeoutError, OSError, websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                    await self.close()
                    if attempt:
                        self._emit_rtt(method, started_at, ok=False)
                        raise
            raise RuntimeError("Falha de comunicacao com a Deriv apos retentativas")  # pragma: no cover

    async def symbols(self) -> list[dict[str, object]]:
        """Le catalogo de instrumentos ativos."""
        result = await self.request({"active_symbols": "brief"})
        return list(result.get("active_symbols") or [])

    async def candles(
        self, symbol: str, count: int = 1000, end: str | int = "latest", period: int = 900
    ) -> list[dict[str, object]]:
        """Le velas da granularidade solicitada, sem subscricao a contratos."""
        if period not in (300, 900, 3600, 86400):
            raise ValueError("periodo nao suportado")
        result = await self.request(
            {"ticks_history": symbol, "style": "candles", "granularity": period, "count": count, "end": end}
        )
        return list(result.get("candles") or [])

    async def close(self) -> None:
        """Fecha conexao publica."""
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
