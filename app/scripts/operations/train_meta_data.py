from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import asyncpg
import numpy as np
from dotenv import load_dotenv

from aether_paths import REPO_ROOT
from src.domain.models.market_data import Candle
from src.infrastructure.api.deriv_pat_binding import DerivPatBindingError, discover_app_id_for_pat, parse_deriv_pat
from src.infrastructure.api.deriv_pat_session import DerivPatSession
from src.infrastructure.api.deriv_rest_client import DerivRestError
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.history_fetch import fetch_paginated_candle_history, parse_history_fetch_config


logger = logging.getLogger("META_TRAIN")
MIN_OHLC_ROWS = 96


@dataclass(frozen=True)
class OhlcBundle:
    symbol: str
    granularity: int
    closes: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    source: str


async def _timescale_inventory(conn: asyncpg.Connection, symbols: list[str]) -> list[tuple[str, int, int]]:
    rows = await conn.fetch(
        """
        SELECT symbol, granularity, COUNT(*)::int AS total
        FROM ohlc_bars
        WHERE symbol = ANY($1::text[])
        GROUP BY symbol, granularity
        ORDER BY symbol, granularity
        """,
        symbols,
    )
    return [(str(r["symbol"]), int(r["granularity"]), int(r["total"])) for r in rows]


async def _fetch_timescale_rows(
    conn: asyncpg.Connection,
    symbol: str,
    granularity: int,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT close, open, high, low, epoch
        FROM ohlc_bars
        WHERE symbol = $1 AND granularity = $2 AND close IS NOT NULL
        ORDER BY epoch ASC
        """,
        symbol,
        granularity,
    )


def _rows_to_bundle(symbol: str, granularity: int, rows: list[asyncpg.Record], *, source: str) -> OhlcBundle | None:
    if len(rows) < MIN_OHLC_ROWS:
        return None
    closes = np.asarray([float(r["close"]) for r in rows], dtype=np.float64)
    open_ = np.asarray([float(r["open"] or r["close"]) for r in rows], dtype=np.float64)
    high = np.asarray([float(r["high"] or r["close"]) for r in rows], dtype=np.float64)
    low = np.asarray([float(r["low"] or r["close"]) for r in rows], dtype=np.float64)
    return OhlcBundle(
        symbol=symbol, granularity=granularity, closes=closes, open_=open_, high=high, low=low, source=source
    )


async def load_bundles_from_timescale(
    dsn: str,
    symbols: list[str],
    granularities: list[int],
) -> list[OhlcBundle]:
    conn = await asyncpg.connect(dsn)
    try:
        bundles: list[OhlcBundle] = []
        for symbol in symbols:
            for granularity in granularities:
                rows = await _fetch_timescale_rows(conn, symbol, granularity)
                bundle = _rows_to_bundle(symbol, granularity, rows, source="timescale")
                if bundle is not None:
                    bundles.append(bundle)
                    break
        return bundles
    finally:
        await conn.close()


def _candles_to_bundle(symbol: str, granularity: int, candles: list[Candle], *, source: str) -> OhlcBundle | None:
    if len(candles) < MIN_OHLC_ROWS:
        return None
    ordered = sorted(candles, key=lambda c: c.epoch)
    closes = np.asarray([float(c.close) for c in ordered], dtype=np.float64)
    open_ = np.asarray([float(c.open) for c in ordered], dtype=np.float64)
    high = np.asarray([float(c.high) for c in ordered], dtype=np.float64)
    low = np.asarray([float(c.low) for c in ordered], dtype=np.float64)
    return OhlcBundle(
        symbol=symbol, granularity=granularity, closes=closes, open_=open_, high=high, low=low, source=source
    )


async def _open_deriv_ws(_settings: dict[str, Any]) -> WebSocketManager:
    load_dotenv(REPO_ROOT / ".env")
    raw_pat = (os.getenv("AETHER_DERIV_PAT") or "").strip()
    if not raw_pat:
        raise RuntimeError("TimescaleDB vazio e AETHER_DERIV_PAT ausente no .env para fallback Deriv.")
    token, _ = parse_deriv_pat(raw_pat)
    try:
        app_id = discover_app_id_for_pat(token, REPO_ROOT)
    except DerivPatBindingError as exc:
        raise RuntimeError(
            "TimescaleDB vazio e App ID Deriv nao encontrado. Rode deriv_pat_connect.py antes do treino."
        ) from exc
    session = DerivPatSession(token, app_id=app_id)
    try:
        result = await session.bootstrap(persist_binding=False)
    except DerivRestError as exc:
        raise RuntimeError(f"Falha ao autenticar Deriv para fallback de historico: {exc}") from exc
    ws = WebSocketManager(result.ws_url, request_timeout=int(session.timeout_seconds))
    await ws.connect()
    return ws


async def load_bundles_from_deriv(
    settings: dict[str, Any],
    symbols: list[str],
    granularity: int,
    bars: int,
) -> list[OhlcBundle]:
    data_cfg = settings.get("data_handler", {}) if isinstance(settings.get("data_handler"), dict) else {}
    fetch_cfg = parse_history_fetch_config(data_cfg if isinstance(data_cfg, dict) else {})
    target = max(MIN_OHLC_ROWS, int(bars))
    ws = await _open_deriv_ws(settings)
    bundles: list[OhlcBundle] = []
    try:
        for symbol in symbols:
            candles = await fetch_paginated_candle_history(
                ws,
                symbol=str(symbol),
                granularity=int(granularity),
                target=target,
                fetch_cfg=fetch_cfg,
                logger=logger,
            )
            bundle = _candles_to_bundle(str(symbol), int(granularity), candles, source="deriv")
            if bundle is not None:
                bundles.append(bundle)
                logger.info("META_TRAIN: %s | %d velas via Deriv (%ds)", symbol, len(bundle.closes), granularity)
    finally:
        await ws.close()
    return bundles


def _granularity_candidates(settings: dict[str, Any], preferred: int) -> list[int]:
    data_cfg = settings.get("data_handler", {}) if isinstance(settings.get("data_handler"), dict) else {}
    micro = int(data_cfg.get("micro_granularity", 60)) if isinstance(data_cfg, dict) else 60
    macro = int(data_cfg.get("granularity", 900)) if isinstance(data_cfg, dict) else 900
    ordered = [int(preferred), micro, macro, 60, 900]
    unique: list[int] = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    return unique


async def _timescale_error(dsn: str, symbols: list[str], granularities: list[int]) -> str:
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:
        return f"Nao foi possivel conectar ao TimescaleDB ({dsn}): {exc}"
    try:
        inventory = await _timescale_inventory(conn, symbols)
    finally:
        await conn.close()
    if not inventory:
        return (
            "TimescaleDB conectado, porem tabela ohlc_bars vazia. "
            f"Tentou granularidades {granularities}. "
            "Rode o motor com infra.enabled=true ou use fallback Deriv (AETHER_DERIV_PAT)."
        )
    lines = [f"{sym}@{gran}s={total}" for sym, gran, total in inventory]
    return f"TimescaleDB sem barras suficientes (min {MIN_OHLC_ROWS}). Inventario: {', '.join(lines)}"


async def resolve_training_bundles(
    *,
    settings: dict[str, Any],
    dsn: str,
    symbols: list[str],
    granularity: int,
    bars: int,
    source: str,
) -> list[OhlcBundle]:
    granularities = _granularity_candidates(settings, granularity)
    mode = str(source or "auto").lower()
    bundles: list[OhlcBundle] = []
    if mode in {"auto", "timescale"}:
        bundles = await load_bundles_from_timescale(dsn, symbols, granularities)
    if bundles:
        return bundles
    if mode == "timescale":
        detail = await _timescale_error(dsn, symbols, granularities)
        raise RuntimeError(detail)
    logger.warning("META_TRAIN: TimescaleDB sem dados; buscando historico na API Deriv.")
    bundles = await load_bundles_from_deriv(settings, symbols, granularity, bars)
    if bundles:
        return bundles
    detail = await _timescale_error(dsn, symbols, granularities)
    raise RuntimeError(
        f"Nenhum dado OHLC disponivel para treino do meta-classificador. {detail} "
        f"Fallback Deriv tambem falhou para granularidade {granularity}s e alvo {bars} barras."
    )
