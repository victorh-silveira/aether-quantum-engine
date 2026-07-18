from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
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
META_TRAIN_DEFAULT_BARS = 5000
META_TRAIN_MAX_BARS = 5000


def resolve_meta_train_bars(bars: int) -> int:
    """Normaliza alvo de barras micro respeitando o teto da API ticks_history."""
    target = max(MIN_OHLC_ROWS, int(bars))
    return min(target, META_TRAIN_MAX_BARS)


@dataclass(frozen=True)
class OhlcBundle:
    symbol: str
    granularity: int
    closes: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    epochs: np.ndarray
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
    max_bars: int | None = None,
) -> list[asyncpg.Record]:
    if max_bars is not None and int(max_bars) > 0:
        return await conn.fetch(
            """
            SELECT close, open, high, low, epoch
            FROM (
                SELECT close, open, high, low, epoch
                FROM ohlc_bars
                WHERE symbol = $1 AND granularity = $2 AND close IS NOT NULL
                ORDER BY epoch DESC
                LIMIT $3
            ) recent
            ORDER BY epoch ASC
            """,
            symbol,
            granularity,
            int(max_bars),
        )
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
    epochs = np.asarray([int(r["epoch"]) for r in rows], dtype=np.int64)
    return OhlcBundle(
        symbol=symbol,
        granularity=granularity,
        closes=closes,
        open_=open_,
        high=high,
        low=low,
        epochs=epochs,
        source=source,
    )


async def load_bundles_from_timescale(
    dsn: str,
    symbols: list[str],
    granularities: list[int],
    max_bars: int = META_TRAIN_DEFAULT_BARS,
) -> list[OhlcBundle]:
    conn = await asyncpg.connect(dsn)
    try:
        bundles: list[OhlcBundle] = []
        bar_target = resolve_meta_train_bars(max_bars)
        for symbol in symbols:
            for granularity in granularities:
                rows = await _fetch_timescale_rows(conn, symbol, granularity, bar_target)
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
    epochs = np.asarray([int(c.epoch) for c in ordered], dtype=np.int64)
    return OhlcBundle(
        symbol=symbol,
        granularity=granularity,
        closes=closes,
        open_=open_,
        high=high,
        low=low,
        epochs=epochs,
        source=source,
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
    target = resolve_meta_train_bars(bars)
    min_complete = max(MIN_OHLC_ROWS, int(target * 0.80))
    symbol_delay = float(fetch_cfg["symbol_delay"])
    ws = await _open_deriv_ws(settings)
    bundles: list[OhlcBundle] = []
    try:
        for index, symbol in enumerate(symbols):
            candles = await fetch_paginated_candle_history(
                ws,
                symbol=str(symbol),
                granularity=int(granularity),
                target=target,
                fetch_cfg=fetch_cfg,
                logger=logger,
            )
            if len(candles) < min_complete:
                logger.warning(
                    "META_TRAIN: %s incompleto (%d/%d); retomando paginacao apos pausa.",
                    symbol,
                    len(candles),
                    target,
                )
                if symbol_delay > 0:
                    await asyncio.sleep(symbol_delay)
                candles = await fetch_paginated_candle_history(
                    ws,
                    symbol=str(symbol),
                    granularity=int(granularity),
                    target=target,
                    fetch_cfg=fetch_cfg,
                    logger=logger,
                    existing=candles,
                )
            bundle = _candles_to_bundle(str(symbol), int(granularity), candles, source="deriv")
            if bundle is not None:
                bundles.append(bundle)
                logger.info("META_TRAIN: %s | %d velas via Deriv (%ds)", symbol, len(bundle.closes), granularity)
            if symbol_delay > 0 and index + 1 < len(symbols):
                await asyncio.sleep(symbol_delay)
    finally:
        await ws.close()
    return bundles


def _granularity_candidates(
    settings: dict[str, Any],
    preferred: int,
    *,
    require_exact: bool = False,
) -> list[int]:
    if require_exact:
        return [int(preferred)]
    data_cfg = settings.get("data_handler", {}) if isinstance(settings.get("data_handler"), dict) else {}
    micro = int(data_cfg.get("micro_granularity", 120)) if isinstance(data_cfg, dict) else 120
    macro = int(data_cfg.get("granularity", 600)) if isinstance(data_cfg, dict) else 600
    ordered = [int(preferred), micro, macro, 120, 600]
    unique: list[int] = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    return unique


def assert_bundles_match_granularity(bundles: list[OhlcBundle], expected: int) -> None:
    expected_gran = int(expected)
    mismatches = [f"{b.symbol}@{b.granularity}" for b in bundles if int(b.granularity) != expected_gran]
    if not mismatches:
        return
    raise RuntimeError(
        f"Granularidade efetiva diverge do alvo {expected_gran}s: {', '.join(mismatches)}. "
        "Retreine com --granularity alinhado ao micro runtime ou popule Timescale na gran correta."
    )


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


async def persist_bundles_to_timescale(dsn: str, bundles: list[OhlcBundle]) -> int:
    if not bundles:
        return 0
    conn = await asyncpg.connect(dsn)
    written = 0
    try:
        for bundle in bundles:
            for idx in range(len(bundle.closes)):
                epoch = int(bundle.epochs[idx])
                ts = datetime.fromtimestamp(epoch, tz=UTC)
                await conn.execute(
                    """
                    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT DO NOTHING
                    """,
                    ts,
                    str(bundle.symbol),
                    epoch,
                    int(bundle.granularity),
                    float(bundle.open_[idx]),
                    float(bundle.high[idx]),
                    float(bundle.low[idx]),
                    float(bundle.closes[idx]),
                )
                written += 1
    finally:
        await conn.close()
    return written


async def resolve_training_bundles(
    *,
    settings: dict[str, Any],
    dsn: str,
    symbols: list[str],
    granularity: int,
    bars: int,
    source: str,
    require_exact_granularity: bool = True,
    seed_timescale_on_deriv: bool = True,
) -> list[OhlcBundle]:
    granularities = _granularity_candidates(
        settings,
        granularity,
        require_exact=bool(require_exact_granularity),
    )
    mode = str(source or "auto").lower()
    bar_target = resolve_meta_train_bars(bars)
    bundles: list[OhlcBundle] = []
    if mode in {"auto", "timescale"}:
        bundles = await load_bundles_from_timescale(dsn, symbols, granularities, bar_target)
        if bundles and require_exact_granularity:
            try:
                assert_bundles_match_granularity(bundles, granularity)
            except RuntimeError:
                bundles = []
    if bundles:
        return bundles
    if mode == "timescale":
        detail = await _timescale_error(dsn, symbols, granularities)
        raise RuntimeError(detail)
    logger.warning(
        "META_TRAIN: TimescaleDB sem dados em %ds; buscando historico na API Deriv.",
        int(granularity),
    )
    bundles = await load_bundles_from_deriv(settings, symbols, granularity, bar_target)
    if bundles:
        assert_bundles_match_granularity(bundles, granularity)
        if seed_timescale_on_deriv:
            try:
                written = await persist_bundles_to_timescale(dsn, bundles)
                logger.info("META_TRAIN: Timescale seed | %d barras gravadas @%ds", written, int(granularity))
            except Exception as exc:
                logger.warning("META_TRAIN: falha ao popular Timescale apos Deriv: %s", exc)
        return bundles
    detail = await _timescale_error(dsn, symbols, granularities)
    raise RuntimeError(
        f"Nenhum dado OHLC disponivel para treino do meta-classificador. {detail} "
        f"Fallback Deriv tambem falhou para granularidade {granularity}s e alvo {bar_target} barras."
    )
