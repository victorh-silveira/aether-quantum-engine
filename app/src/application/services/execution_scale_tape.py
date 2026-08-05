"""Helpers de fita OHLC (velas anterior/atual) e consenso multi-escala."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.models.trade import TradeDirection


def bar_direction_at(
    opens: np.ndarray | list[float] | None,
    closes: np.ndarray | list[float] | None,
    *,
    offset: int = -1,
) -> str | None:
    """Direcao CALL/PUT de uma vela por offset (-1 atual, -2 anterior); None se flat/ausente."""
    o_arr = np.asarray(opens if opens is not None else [], dtype=np.float64).reshape(-1)
    c_arr = np.asarray(closes if closes is not None else [], dtype=np.float64).reshape(-1)
    n = min(int(o_arr.size), int(c_arr.size))
    if n < 1:
        return None
    idx = int(offset) if int(offset) >= 0 else n + int(offset)
    if idx < 0 or idx >= n:
        return None
    open_v = float(o_arr[idx])
    close_v = float(c_arr[idx])
    delta = close_v - open_v
    if abs(delta) <= 1e-12:
        return None
    return TradeDirection.CALL.name if delta > 0.0 else TradeDirection.PUT.name


def last_bar_direction(
    opens: np.ndarray | list[float] | None,
    closes: np.ndarray | list[float] | None,
) -> str | None:
    """Direcao CALL/PUT da vela atual (ultimo OHLC do buffer, em formacao)."""
    return bar_direction_at(opens, closes, offset=-1)


def prev_bar_direction(
    opens: np.ndarray | list[float] | None,
    closes: np.ndarray | list[float] | None,
) -> str | None:
    """Direcao CALL/PUT da vela anterior fechada (penultimo OHLC do buffer)."""
    return bar_direction_at(opens, closes, offset=-2)


def tape_consensus(dirs: list[str | None], *, min_votes: int = 2) -> str | None:
    """Maioria CALL/PUT entre votos validos; None se empate ou votos insuficientes."""
    need = max(1, int(min_votes))
    votes = [d for d in dirs if d in {TradeDirection.CALL.name, TradeDirection.PUT.name}]
    if len(votes) < need:
        return None
    call_n = sum(1 for d in votes if d == TradeDirection.CALL.name)
    put_n = len(votes) - call_n
    if call_n >= need and call_n > put_n:
        return TradeDirection.CALL.name
    if put_n >= need and put_n > call_n:
        return TradeDirection.PUT.name
    return None


def mini_bar_pair_agrees(metrics: dict[str, Any], consensus: str) -> bool:
    """True se vela MINI anterior e atual existem e batem com o consenso."""
    side = str(consensus or "").upper()
    if side not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return False
    prev = str(metrics.get("scale_mini_prev_bar_dir") or "").upper()
    curr = str(metrics.get("scale_mini_bar_dir") or "").upper()
    return prev == side and curr == side


def compute_tape_strong(
    metrics: dict[str, Any],
    consensus: str | None,
    *,
    mini_pair_sufficient: bool = True,
) -> bool:
    """Fita forte: par MINI alinhado; MILI/MICRO reforcam se mini_pair_sufficient=False."""
    side = str(consensus or "").upper()
    if not mini_bar_pair_agrees(metrics, side):
        return False
    if mini_pair_sufficient:
        return True
    mili = str(metrics.get("scale_mili_dir") or "").upper()
    if mili == side:
        return True
    mc_prev = str(metrics.get("scale_micro_prev_bar_dir") or "").upper()
    mc_curr = str(metrics.get("scale_micro_bar_dir") or "").upper()
    return mc_prev == side and mc_curr == side


def mini_pair_opposes_tcn(metrics: dict[str, Any], tcn_dir: str | None) -> bool:
    """True se par MINI unanime existe e discrepa do lado TCN."""
    side = str(tcn_dir or "").upper()
    if side not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return False
    prev = str(metrics.get("scale_mini_prev_bar_dir") or "").upper()
    curr = str(metrics.get("scale_mini_bar_dir") or "").upper()
    if prev not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return False
    if prev != curr:
        return False
    return prev != side
