"""Adaptador de dados de mercado financeiro real para o subsistema indicador."""

from __future__ import annotations

import logging

import httpx

from indicator.domain import Candle

logger = logging.getLogger(__name__)

REAL_MARKET_SYMBOL_MAP: dict[str, str] = {
    "OTC_SPC": "^GSPC",
    "OTC_NDX": "^NDX",
    "OTC_DJI": "^DJI",
    "OTC_GDAXI": "^GDAXI",
    "OTC_FTSE": "^FTSE",
    "OTC_N225": "^N225",
    "OTC_SX5E": "^STOXX50E",
    "OTC_SSMI": "^SSMI",
    "OTC_FCHI": "^FCHI",
    "OTC_AEX": "^AEX",
    "OTC_HSI": "^HSI",
    "OTC_AS51": "^AXJO",
}

INTERVAL_MAP: dict[int, str] = {
    60: "1m",
    120: "2m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "60m",
    86400: "1d",
}


def resolve_ticker(symbol: str) -> str | None:
    """Retorna o ticker oficial do mercado para o ativo derivativo."""
    return REAL_MARKET_SYMBOL_MAP.get(symbol)


async def fetch_real_candles(
    symbol: str,
    period_seconds: int = 900,
    range_str: str = "5d",
    timeout: float = 10.0,
) -> list[Candle]:
    """Busca velas de mercado real e converte para o modelo de Candle do indicador."""
    ticker = resolve_ticker(symbol)
    if not ticker:
        return []
    interval = INTERVAL_MAP.get(period_seconds, "15m")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_str}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("Falha Yahoo Finance para %s: status %d", symbol, resp.status_code)
                return []
            result = resp.json().get("chart", {}).get("result", [])
            if not result:
                return []
            chart_data = result[0]
            timestamps = chart_data.get("timestamp", [])
            quotes = chart_data.get("indicators", {}).get("quote", [{}])[0]
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            candles = []
            for i, ts in enumerate(timestamps):
                if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
                    break
                o, h, lo, c = opens[i], highs[i], lows[i], closes[i]
                if any(v is None for v in (o, h, lo, c)):
                    continue
                candles.append(
                    Candle(
                        epoch=int(ts),
                        open=float(o),
                        high=float(h),
                        low=float(lo),
                        close=float(c),
                    )
                )
            return candles
    except Exception as exc:
        logger.error("Erro ao buscar dados reais para %s: %s", symbol, exc)
        return []
