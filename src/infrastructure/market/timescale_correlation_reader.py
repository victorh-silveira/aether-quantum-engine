"""Leitura de matriz de correlacao cruzada via TimescaleDB."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import asyncpg
import numpy as np


logger = logging.getLogger("AETH")


def _log_returns(closes: list[float]) -> np.ndarray:
    """Retornos logaritmicos a partir de precos de fechamento."""
    arr = np.asarray(closes, dtype=np.float64)
    if arr.size < 2:
        return np.array([], dtype=np.float64)
    prev = arr[:-1]
    nxt = arr[1:]
    mask = (prev > 0.0) & (nxt > 0.0)
    if not mask.any():
        return np.array([], dtype=np.float64)
    return np.log(nxt[mask] / prev[mask])


def compute_correlation_matrix(closes_by_symbol: dict[str, list[float]]) -> dict[tuple[str, str], float]:
    """Calcula correlacao de Pearson entre pares de simbolos."""
    symbols = sorted(closes_by_symbol.keys())
    returns: dict[str, np.ndarray] = {}
    for sym in symbols:
        rets = _log_returns(closes_by_symbol[sym])
        if rets.size >= 3:
            returns[sym] = rets
    matrix: dict[tuple[str, str], float] = {}
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i:]:
            if sym_a not in returns or sym_b not in returns:
                corr = 1.0 if sym_a == sym_b else 0.0
            else:
                a = returns[sym_a]
                b = returns[sym_b]
                n = min(a.size, b.size)
                corr = float(np.corrcoef(a[-n:], b[-n:])[0, 1])
                if not math.isfinite(corr):
                    corr = 0.0
            matrix[(sym_a, sym_b)] = corr
            matrix[(sym_b, sym_a)] = corr
    return matrix


async def fetch_symbol_closes(
    dsn: str,
    symbols: list[str],
    *,
    granularity: int,
    bars: int,
) -> dict[str, list[float]]:
    """Busca ultimos fechamentos OHLC por simbolo no TimescaleDB."""
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        out: dict[str, list[float]] = {}
        async with pool.acquire() as conn:
            for symbol in symbols:
                rows = await conn.fetch(
                    """
                    SELECT close
                    FROM ohlc_bars
                    WHERE symbol = $1 AND granularity = $2 AND close IS NOT NULL
                    ORDER BY time DESC
                    LIMIT $3
                    """,
                    str(symbol),
                    int(granularity),
                    int(bars),
                )
                closes = [float(row["close"]) for row in reversed(rows)]
                out[str(symbol)] = closes
        return out
    finally:
        await pool.close()


async def fetch_correlation_matrix(
    dsn: str,
    symbols: list[str],
    *,
    granularity: int = 60,
    bars: int = 120,
) -> dict[tuple[str, str], float]:
    """Obtem matriz de correlacao cruzada a partir do TimescaleDB."""
    closes = await fetch_symbol_closes(dsn, symbols, granularity=granularity, bars=bars)
    return compute_correlation_matrix(closes)


def correlation_matrix_from_cache(raw: str | bytes | None) -> dict[tuple[str, str], float]:
    """Deserializa matriz de correlacao armazenada em Redis."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    matrix: dict[tuple[str, str], float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or "|" not in key:
            continue
        left, right = key.split("|", 1)
        matrix[(left, right)] = float(value)
    return matrix


def correlation_matrix_to_cache(matrix: dict[tuple[str, str], float]) -> str:
    """Serializa matriz de correlacao para Redis."""
    payload = {f"{a}|{b}": float(v) for (a, b), v in matrix.items()}
    return json.dumps(payload)


async def read_cached_correlation_matrix(orch: Any) -> dict[tuple[str, str], float]:
    """Le matriz de correlacao do cache Redis do orquestrador."""
    store = getattr(orch, "state_store", None)
    if store is None:
        return {}
    getter = getattr(store, "get_string", None)
    if not callable(getter):
        return {}
    raw = await getter("corr_matrix")
    return correlation_matrix_from_cache(raw)
