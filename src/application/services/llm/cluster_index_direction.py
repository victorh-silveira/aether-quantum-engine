"""Direcao CALL/PUT por indice OTC a partir de movimento RISE/FALL no snapshot."""

from __future__ import annotations

import re


def move_token_from_cluster_part(part: str) -> str | None:
    """Extrai RISE, FALL ou FLAT de uma linha de indice do cluster."""
    m = re.search(r"\((RISE|FALL|FLAT)\s", str(part or "").upper())
    if not m:
        return None
    return m.group(1)


def index_trade_direction_from_move(move: str, *, mode: str = "counter_trend") -> str | None:
    """Mapeia movimento realtime do indice para CALL/PUT em contratos RISE_FALL."""
    token = str(move or "").upper()
    if token == "FLAT":
        return None
    if mode == "momentum":
        return "CALL" if token == "RISE" else "PUT" if token == "FALL" else None
    if token == "FALL":
        return "CALL"
    if token == "RISE":
        return "PUT"
    return None


def build_cluster_index_directions(
    us_symbols: list[str],
    eu_symbols: list[str],
    us_parts: tuple[str, ...],
    eu_parts: tuple[str, ...],
    *,
    mode: str = "counter_trend",
) -> dict[str, str]:
    """Monta direcao CALL/PUT por simbolo de indice a partir das partes do snapshot."""
    out: dict[str, str] = {}
    for sym, part in zip(us_symbols, us_parts, strict=False):
        move = move_token_from_cluster_part(part)
        if not move:
            continue
        tag = index_trade_direction_from_move(move, mode=mode)
        if tag:
            out[str(sym)] = tag
    for sym, part in zip(eu_symbols, eu_parts, strict=False):
        move = move_token_from_cluster_part(part)
        if not move:
            continue
        tag = index_trade_direction_from_move(move, mode=mode)
        if tag:
            out[str(sym)] = tag
    return out
