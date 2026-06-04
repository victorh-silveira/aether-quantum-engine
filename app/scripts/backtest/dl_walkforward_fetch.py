"""Download de velas M1 da Deriv para backtest Deep Learning."""

from typing import Any

from src.application.services.auth_manager import AuthManager
from src.infrastructure.api.websocket_manager import WebSocketManager


M1_GRANULARITY = 60


async def fetch_m1_closes(symbol: str, count: int, config: dict[str, Any]) -> list[float]:
    api = config.get("api_config", {})
    ws = WebSocketManager(
        api.get("base_url", "wss://ws.derivws.com/websockets/v3?app_id=1089"),
        request_timeout=int(api.get("request_timeout_seconds", 60)),
    )
    mode = str(config.get("trading", {}).get("mode", "demo"))
    token = AuthManager(mode=mode).get_token()
    if not token:
        raise RuntimeError("Token Deriv ausente no .env")
    await ws.connect()
    try:
        auth = await ws.send({"authorize": token})
        if isinstance(auth, dict) and auth.get("error"):
            raise RuntimeError(f"Deriv authorize falhou: {auth.get('error')}")
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
