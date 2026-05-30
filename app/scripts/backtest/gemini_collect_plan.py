"""Plano e estimativa de consultas Gemini no backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.backtest.gemini_cache import load_cache
from scripts.backtest.gemini_schedule import gemini_query_points, schedule_label


def estimate_gemini_minutes(query_count: int, *, cached: int, seconds_per_call: float = 25.0) -> float:
    """Estimativa grossa de duracao (chamadas novas x latencia media)."""
    pending = max(0, query_count - cached)
    return (pending * seconds_per_call) / 60.0


def print_gemini_plan(
    *,
    start: int,
    end: int,
    schedule: str,
    step: int,
    cache_path: Path | None,
    max_llm_bars: int | None,
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    macro_cfg: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
    anchor: str | None = None,
) -> list[int]:
    """Exibe plano no terminal e retorna pontos de consulta API."""
    targets = gemini_query_points(
        start,
        end,
        schedule,
        step,
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        macro_cfg=macro_cfg,
        config=config,
        anchor=anchor,
    )
    if max_llm_bars is not None:
        targets = targets[: max(0, int(max_llm_bars))]
    cached = 0
    if cache_path and cache_path.is_file():
        keys = load_cache(cache_path)
        cached = sum(1 for b in targets if str(b) in keys)
    pending = len(targets) - cached
    eta_min = estimate_gemini_minutes(len(targets), cached=cached)
    print(
        f"Gemini ({schedule_label(schedule, config=config)}): {len(targets)} consultas API"
        f" | {end - start + 1} velas primarias simuladas"
        f" | cache: {cached} prontas, {pending} pendentes | ~{eta_min:.0f} min",
        flush=True,
    )
    return targets
