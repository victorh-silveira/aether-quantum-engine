"""Direcao M5 por indice e mapa para correcao de cluster em divergencia."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.global_macro_confluence import cluster_direction_from_closes
from src.application.services.llm.macro_config import resolve_macro_config


def index_m5_direction_from_closes(
    closes: list[float],
    *,
    threshold_pct: float,
    min_move_pct: float,
) -> str:
    return cluster_direction_from_closes(closes, threshold_pct, min_move_pct)


def build_index_m5_dir_map(
    closes_by_symbol: dict[str, list[float]],
    macro_cfg: dict[str, Any] | None,
) -> dict[str, str]:
    cfg = resolve_macro_config(macro_cfg)
    threshold = float(cfg["cluster_return_threshold_pct"])
    min_move = float(cfg["cluster_fallback_min_move_pct"])
    out: dict[str, str] = {}
    for sym, closes in closes_by_symbol.items():
        if not isinstance(closes, list) or len(closes) < 2:
            continue
        out[str(sym)] = index_m5_direction_from_closes(
            closes,
            threshold_pct=threshold,
            min_move_pct=min_move,
        )
    return out
