"""Adapter de market data publico da Deriv, sem autenticacao de conta."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

PUBLIC_URL = "wss://api.derivws.com/trading/v1/options/ws/public"


class PublicDeriv:
    """Cliente estritamente limitado a catalogo e historico OHLC."""

    def __init__(
        self, url: str = PUBLIC_URL, request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
    ) -> None:
        self.url = url
        self._request_override = request
        self._socket: Any = None
        self._lock = asyncio.Lock()
        self._request_id = 0

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Envia requisicao publica autorizada com reconexao unica."""
        if set(payload) - {"active_symbols", "ticks_history", "style", "granularity", "count", "end"}:
            raise ValueError("operacao nao permitida no indicador")
        if self._request_override is not None:
            return await self._request_override(payload)
        async with self._lock:
            for attempt in range(2):
                try:
                    if self._socket is None:
                        self._socket = await websockets.connect(self.url, open_timeout=15, max_size=8_000_000)
                    self._request_id += 1
                    await self._socket.send(json.dumps({**payload, "req_id": self._request_id}))
                    response = json.loads(await asyncio.wait_for(self._socket.recv(), timeout=20))
                    if response.get("error"):
                        raise RuntimeError(str(response["error"].get("message", response["error"])))
                    if response.get("req_id") not in (None, self._request_id):
                        raise RuntimeError("resposta fora de ordem")
                    return response
                except (TimeoutError, OSError, websockets.exceptions.ConnectionClosed):
                    await self.close()
                    if attempt:
                        raise

    async def symbols(self) -> list[dict[str, object]]:
        """Le catalogo de instrumentos ativos."""
        result = await self.request({"active_symbols": "brief"})
        return list(result.get("active_symbols") or [])

    async def candles(self, symbol: str, count: int = 1000, end: str | int = "latest") -> list[dict[str, object]]:
        """Le velas M5, sem subscricao a contratos."""
        result = await self.request(
            {"ticks_history": symbol, "style": "candles", "granularity": 300, "count": count, "end": end}
        )
        return list(result.get("candles") or [])

    async def close(self) -> None:
        """Fecha conexao publica."""
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
