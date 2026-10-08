"""Politica de barras meta-ready para seed Timescale (micro vs D1)."""

from __future__ import annotations

MIN_BARS_MICRO = 5000
MIN_BARS_MACRO_D1 = 365
MACRO_D1_GRANULARITY = 86400


def min_bars_for_granularity(granularity: int) -> int:
    """Piso meta-ready: micro M5=5000; macro D1 (>=86400)=365."""
    if int(granularity) >= MACRO_D1_GRANULARITY:
        return MIN_BARS_MACRO_D1
    return MIN_BARS_MICRO


def resolve_seed_bars_for_granularity(granularity: int, bars_cap: int | None = None) -> int:
    """Barras a buscar no seed: D1 sempre 365; micro respeita piso/teto 5000."""
    floor = min_bars_for_granularity(granularity)
    if int(granularity) >= MACRO_D1_GRANULARITY:
        return floor
    if bars_cap is None:
        return floor
    return min(max(int(bars_cap), floor), MIN_BARS_MICRO)
