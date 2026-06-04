"""Download de velas M1 da Deriv para backtest Deep Learning."""

from typing import Any

from src.infrastructure.api.websocket_manager import WebSocketManager


M1_GRANULARITY = 60


async def fetch_m1_closes(symbol: str, count: int, config: dict[str, Any]) -> list[float]:
    api = config.get("api_config", {})
    public_ws = api.get(
        "public_ws_url",
        "wss://api.derivws.com/trading/v1/options/ws/public",
    )
    ws = WebSocketManager(public_ws, request_timeout=int(api.get("request_timeout_seconds", 60)))
    await ws.connect()
    try:
        req = {
            "ticks_history": symbol,
            "end": "latest",
            "style": "candles",
            "granularity": M1_GRANULARITY,
            "count": count,
        }
        res = await ws.send(req)
        candles = res.get("candles") if isinstance(res, dict) else []
        out: list[float] = []
        for candle in candles or []:
            if isinstance(candle, dict):
                out.append(float(candle["close"]))
        return out
    finally:
        await ws.close()
