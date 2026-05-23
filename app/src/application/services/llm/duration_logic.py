"""Logica para calculo de duracao adaptativa baseada em regime e volatilidade."""

from __future__ import annotations

from typing import Any


def calculate_adaptive_duration(
    _runtime: dict[str, Any],
    ctx: dict[str, Any],
    base_duration: int | str = 1,
) -> int | str:
    """Retorna duracao otimizada (em minutos) baseada no regime e volatilidade."""
    if base_duration == "MULT":
        return "MULT"

    regime = str(ctx.get("regime_label", "range")).lower()
    atr = float(ctx.get("atr_m5_pct") or 0.0)

    duration = int(base_duration)

    if duration == 1:
        return duration

    if regime in ("trend_fraca", "range"):
        duration = max(duration, 2)

    if atr > 0.40:
        duration = max(duration, 5)
    elif atr > 0.25:
        duration = max(duration, 3)
    elif atr > 0.18:
        duration = max(duration, 2)

    return int(duration)


def enforce_minimum_duration(symbol: str, current_duration: int | str) -> int | str:
    """Garante que simbolos respeitem a duracao minima (vazio se nao houver restricoes)."""
    _ = symbol
    return current_duration
