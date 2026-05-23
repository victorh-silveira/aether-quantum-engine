"""Carga de velas M15/M5 da Deriv para backtest Medallion (sempre fetch fresco)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.application.services.auth_manager import AuthManager
from src.application.services.llm.strategy_clusters import resolve_cluster_lists
from src.infrastructure.api.websocket_manager import WebSocketManager


M15_GRANULARITY = 900
M5_GRANULARITY = 300


def load_settings(path: Path) -> dict[str, Any]:
    """Carrega settings.json do projeto."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def backtest_symbols(config: dict[str, Any]) -> tuple[list[str], list[str], list[str], str]:
    """Retorna listas US, EU, todos os simbolos e ancora."""
    strategy = config.get("strategy", {})
    us_syms, eu_syms = resolve_cluster_lists(strategy if isinstance(strategy, dict) else None)
    anchor = str(config.get("anchor", "frxEURUSD"))
    all_syms = list(dict.fromkeys([anchor, *us_syms, *eu_syms]))
    return us_syms, eu_syms, all_syms, anchor


def _parse_candle_closes(res: dict[str, Any] | None) -> list[float]:
    """Extrai fechamentos de uma resposta ticks_history."""
    if not isinstance(res, dict) or res.get("error"):
        return []
    history = res.get("candles") or []
    out: list[float] = []
    for candle in history:
        if not isinstance(candle, dict):
            continue
        try:
            out.append(float(candle["close"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def _fetch_closes(
    ws: WebSocketManager,
    symbol: str,
    granularity: int,
    count: int,
    *,
    retries: int = 3,
) -> list[float]:
    """Busca fechamentos OHLC historicos via ticks_history com retentativas."""
    if count <= 0:
        return []
    req = {
        "ticks_history": symbol,
        "end": "latest",
        "style": "candles",
        "granularity": granularity,
        "count": count,
    }
    last: list[float] = []
    for attempt in range(max(1, retries)):
        try:
            res = await ws.send(req)
        except Exception:
            res = None
        last = _parse_candle_closes(res if isinstance(res, dict) else None)
        if last:
            return last
        if attempt + 1 < retries:
            await asyncio.sleep(0.4 * (attempt + 1))
    return last


async def fetch_market_data(
    config: dict[str, Any],
    *,
    m15_bars: int,
    m5_bars: int,
) -> dict[str, Any]:
    """Baixa series M15 e M5 para o universo configurado."""
    _, _, all_syms, _ = backtest_symbols(config)
    api = config.get("api_config", {})
    ws = WebSocketManager(
        api.get("base_url", "wss://ws.derivws.com/websockets/v3?app_id=1089"),
        request_timeout=int(api.get("request_timeout_seconds", 60)),
    )
    mode = str(config.get("trading", {}).get("mode", "demo"))
    token = AuthManager(mode=mode).get_token()
    if not token:
        raise RuntimeError("Token Deriv ausente no .env (AETHER_DEMO_TOKEN ou AETHER_LIVE_TOKEN)")

    await ws.connect()
    try:
        auth = await ws.send({"authorize": token})
        if isinstance(auth, dict) and auth.get("error"):
            raise RuntimeError(f"Deriv authorize falhou: {auth.get('error')}")
        m15: dict[str, list[float]] = {}
        m5: dict[str, list[float]] = {}
        for sym in all_syms:
            m15[sym] = await _fetch_closes(ws, sym, M15_GRANULARITY, m15_bars)
            m5[sym] = await _fetch_closes(ws, sym, M5_GRANULARITY, m5_bars)
            await asyncio.sleep(0.2)
        for sym in [s for s in all_syms if not m15.get(s)]:
            await asyncio.sleep(0.5)
            m15[sym] = await _fetch_closes(ws, sym, M15_GRANULARITY, m15_bars, retries=4)
            m5[sym] = await _fetch_closes(ws, sym, M5_GRANULARITY, m5_bars, retries=4)
    finally:
        await ws.close()

    return {
        "m15": m15,
        "m5": m5,
        "meta": {
            "granularity_m15": M15_GRANULARITY,
            "granularity_m5": M5_GRANULARITY,
            "m15_bars": m15_bars,
            "m5_bars": m5_bars,
            "symbols": all_syms,
            "data_source": "deriv_fetch",
        },
    }


def resolve_bar_counts(config: dict[str, Any], *, days: int | None, bars: int | None) -> tuple[int, int]:
    """Calcula quantidade de velas M15 e M5 a buscar."""
    macro = config.get("strategy", {}).get("macro", {})
    if isinstance(macro, dict):
        lookback = max(
            int(macro.get("statarb_lookback", 30)),
            int(macro.get("cluster_bars", 8)),
            30,
        )
    else:
        lookback = 30
    if bars is not None and bars > 0:
        m15_count = bars + lookback + 5
    elif days is not None and days > 0:
        m15_count = days * 96 + lookback + 5
    else:
        m15_count = 14 * 96 + lookback + 5
    m5_count = m15_count * 3 + int(macro.get("cluster_fallback_bars", 12) if isinstance(macro, dict) else 12)
    return m15_count, m5_count


def _align_series_lengths(
    m15: dict[str, list[float]], m5: dict[str, list[float]]
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    """Alinha todas as series M15 ao mesmo comprimento (cauda comum)."""
    if not m15:
        return m15, m5, 0
    lengths = [len(series) for series in m15.values() if series]
    if not lengths:
        return m15, m5, 0
    min_len = min(lengths)
    if min_len <= 0:
        return m15, m5, 0
    m15_out = {sym: series[-min_len:] for sym, series in m15.items()}
    m5_out = {sym: series[-min(min_len * 3, len(series)) :] for sym, series in m5.items()}
    return m15_out, m5_out, min_len


async def fetch_market_for_backtest(
    config: dict[str, Any],
    *,
    days: int | None,
    bars: int | None,
) -> dict[str, Any]:
    """Download fresco na Deriv para o intervalo pedido (sem cache em disco)."""
    m15_bars, m5_bars = resolve_bar_counts(config, days=days, bars=bars)
    payload = await fetch_market_data(config, m15_bars=m15_bars, m5_bars=m5_bars)

    eval_bars = 0
    if bars is not None and bars > 0:
        eval_bars = int(bars)
    elif days is not None and days > 0:
        eval_bars = int(days) * 96

    m15_aligned, m5_aligned, aligned_len = _align_series_lengths(payload["m15"], payload["m5"])
    if aligned_len <= 0:
        empty = [sym for sym, series in payload.get("m15", {}).items() if not series]
        raise RuntimeError(
            "Deriv nao devolveu velas M15 suficientes para o backtest. "
            f"Simbolos vazios ({len(empty)}): {', '.join(empty) or 'todos'}. "
            "Aguarde 10-30s e execute novamente (rate limit da API)."
        )
    meta = dict(payload.get("meta") or {})
    meta.update(
        {
            "window_days_requested": days,
            "window_bars_requested": bars,
            "m15_bars_fetched": m15_bars,
            "m5_bars_fetched": m5_bars,
            "m15_eval_bars_target": eval_bars,
            "m15_bars_aligned": aligned_len,
        }
    )
    return {"m15": m15_aligned, "m5": m5_aligned, "meta": meta}
