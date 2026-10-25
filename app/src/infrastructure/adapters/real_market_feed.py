"""Adaptador de integracao com APIs reais de dados do mercado financeiro."""

from __future__ import annotations

import logging
from typing import Any

import httpx


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


def resolve_real_market_symbol(symbol: str) -> str | None:
    """Mapeia o simbolo da corretora para o ticker oficial do mercado financeiro."""
    return REAL_MARKET_SYMBOL_MAP.get(symbol)


class RealMarketFeed:
    """Cliente HTTP assincrono para obtencao de dados reais de mercado."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }

    async def fetch_real_candles(
        self,
        symbol: str,
        period_seconds: int = 900,
        range_str: str = "5d",
    ) -> list[dict[str, Any]]:
        """Busca series de velas OHLC historicas oficiais do mercado financeiro."""
        ticker = resolve_real_market_symbol(symbol)
        if not ticker:
            return []
        interval = INTERVAL_MAP.get(period_seconds, "15m")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": range_str}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.warning(
                        "Falha ao buscar velas reais para %s (%d): status %d", symbol, period_seconds, resp.status_code
                    )
                    return []
                payload = resp.json()
                result = payload.get("chart", {}).get("result", [])
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
                    op_v, hi_v, lo_v, cl_v = opens[i], highs[i], lows[i], closes[i]
                    if any(v is None for v in (op_v, hi_v, lo_v, cl_v)):
                        continue
                    candles.append(
                        {
                            "epoch": int(ts),
                            "open": float(op_v),
                            "high": float(hi_v),
                            "low": float(lo_v),
                            "close": float(cl_v),
                        }
                    )
                return candles
        except Exception as exc:
            logger.error("Erro na requisicao de dados reais de mercado para %s: %s", symbol, exc)
            return []

    async def fetch_real_quote(self, symbol: str) -> dict[str, Any] | None:
        """Obtem cotacao instantanea oficial mais recente do ativo."""
        ticker = resolve_real_market_symbol(symbol)
        if not ticker:
            return None
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1m", "range": "1d"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    return None
                result = resp.json().get("chart", {}).get("result", [])
                if not result:
                    return None
                meta = result[0].get("meta", {})
                regular_price = meta.get("regularMarketPrice")
                prev_close = meta.get("chartPreviousClose")
                return {
                    "symbol": symbol,
                    "ticker": ticker,
                    "price": float(regular_price) if regular_price is not None else None,
                    "previous_close": float(prev_close) if prev_close is not None else None,
                    "currency": meta.get("currency"),
                    "exchange_timezone": meta.get("exchangeTimezoneName"),
                }
        except Exception as exc:
            logger.error("Erro ao buscar cotacao real de %s: %s", symbol, exc)
            return None
