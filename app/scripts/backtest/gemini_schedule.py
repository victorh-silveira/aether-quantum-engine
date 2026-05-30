"""Agendamento de consultas Gemini no backtest (sem 1 chamada por vela)."""

from __future__ import annotations

from typing import Any

from scripts.backtest.snapshot_engine import build_snapshot_at_bar
from scripts.backtest.timeframe import bars_per_day, primary_granularity_seconds


SCHEDULE_DAILY = "daily"
SCHEDULE_BAR = "bar"
SCHEDULE_TAG_CHANGE = "tag_change"


def gemini_query_points(
    start: int,
    end: int,
    schedule: str,
    step: int,
    *,
    m15: dict[str, list[float]] | None = None,
    m5: dict[str, list[float]] | None = None,
    us_syms: list[str] | None = None,
    eu_syms: list[str] | None = None,
    macro_cfg: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    anchor: str | None = None,
) -> list[int]:
    """Pontos de barra primaria onde a API Gemini e chamada (nao todas as velas)."""
    per_day = bars_per_day(primary_granularity_seconds(config or {}))
    sched = (schedule or SCHEDULE_DAILY).strip().lower()
    if sched == SCHEDULE_DAILY:
        points: list[int] = []
        day_bar = start
        while day_bar <= end:
            points.append(day_bar)
            day_bar += per_day
        return points
    if sched == SCHEDULE_BAR:
        s = max(1, int(step))
        return [i for i in range(start, end + 1) if (i - start) % s == 0]
    if sched == SCHEDULE_TAG_CHANGE:
        if not m15 or not us_syms or not eu_syms:
            return [start]
        points = []
        last_tag: str | None = None
        for bar_index in range(start, end + 1):
            snap = build_snapshot_at_bar(
                bar_index=bar_index,
                m15_closes=m15,
                m5_closes=m5 or {},
                us_symbols=us_syms,
                eu_symbols=eu_syms,
                macro_cfg=macro_cfg,
                anchor=anchor or "",
            )
            if snap.tag != last_tag:
                points.append(bar_index)
                last_tag = snap.tag
        return points or [start]
    return gemini_query_points(start, end, SCHEDULE_DAILY, step)


def payload_for_bar(
    cache: dict[str, dict[str, Any]],
    bar_index: int,
    start: int,
    schedule: str,
    step: int,
    query_points: list[int],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Payload Gemini vigente na barra (replica decisao do ponto de agenda)."""
    per_day = bars_per_day(primary_granularity_seconds(config or {}))
    sched = (schedule or SCHEDULE_DAILY).strip().lower()
    if sched == SCHEDULE_DAILY:
        day_start = start + ((bar_index - start) // per_day) * per_day
        return cache.get(str(day_start))
    if sched == SCHEDULE_BAR:
        if (bar_index - start) % max(1, int(step)) != 0:
            return None
        return cache.get(str(bar_index))
    if sched == SCHEDULE_TAG_CHANGE:
        eligible = [p for p in query_points if p <= bar_index]
        if not eligible:
            return None
        return cache.get(str(max(eligible)))
    return None


def schedule_label(schedule: str, *, config: dict[str, Any] | None = None) -> str:
    """Rotulo legivel para logs."""
    per_day = bars_per_day(primary_granularity_seconds(config or {}))
    sched = (schedule or SCHEDULE_DAILY).strip().lower()
    if sched == SCHEDULE_DAILY:
        return f"1 consulta por dia de sessao ({per_day} velas primarias)"
    if sched == SCHEDULE_BAR:
        return "consulta a cada N velas (--llm-bar-step)"
    if sched == SCHEDULE_TAG_CHANGE:
        return "consulta quando a tag macro muda"
    return sched
