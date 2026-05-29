"""Coleta de ordens HFT no backtest com decisao Gemini (mesmo prompt do live)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from scripts.backtest.gemini_cache import load_cache, save_cache
from scripts.backtest.gemini_collect_api import BacktestOrchestrator, warm_gemini_cache
from scripts.backtest.gemini_collect_hft import collect_hft_with_resolver
from scripts.backtest.gemini_collect_plan import estimate_gemini_minutes, print_gemini_plan
from scripts.backtest.gemini_schedule import SCHEDULE_DAILY, payload_for_bar
from scripts.backtest.historical_stream import HistoricalStream
from scripts.backtest.signal_engine import BacktestOrder, resolve_orders_from_cluster_tags
from src.application.services.llm.context_runtime import resolve_llm_runtime
from src.application.services.llm.symbol_decision_utils import decision_from_payload


__all__ = ("BacktestOrchestrator", "collect_hft_orders_gemini", "estimate_gemini_minutes")


async def collect_hft_orders_gemini(
    *,
    config: dict[str, Any],
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    all_syms: list[str],
    anchor: str,
    macro_cfg: dict[str, Any] | None,
    start: int,
    end: int,
    cache_path: str | None = None,
    max_llm_bars: int | None = None,
    llm_bar_step: int = 5,
    gemini_schedule: str = SCHEDULE_DAILY,
) -> tuple[list[BacktestOrder], dict[str, Any]]:
    """Agenda poucas consultas Gemini; HFT usa a mesma decisao ate o proximo ponto."""
    stream = HistoricalStream(m15, m5, bar_index=start)
    orch = BacktestOrchestrator(config, stream=stream, anchor=anchor, symbols=all_syms)
    runtime = resolve_llm_runtime(orch)
    cache_file = Path(cache_path) if cache_path else None
    cache = load_cache(cache_file) if cache_file else {}
    step = max(1, int(llm_bar_step))
    schedule = (gemini_schedule or SCHEDULE_DAILY).strip().lower()

    targets = print_gemini_plan(
        start=start,
        end=end,
        schedule=schedule,
        step=step,
        cache_path=cache_file,
        max_llm_bars=max_llm_bars,
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        macro_cfg=macro_cfg,
    )
    sys.stdout.flush()

    llm_calls, llm_failures = await warm_gemini_cache(
        orch=orch,
        stream=stream,
        anchor=anchor,
        runtime=runtime,
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        macro_cfg=macro_cfg,
        targets=targets,
        cache=cache,
        cache_file=cache_file,
    )

    if cache_file:
        save_cache(cache_file, cache)

    print("Gemini: simulando HFT (decisao replicada nos pontos agendados)...", flush=True)

    async def _resolver(bar_index: int, snapshot, runtime) -> list[BacktestOrder]:
        payload = payload_for_bar(cache, bar_index, start, schedule, step, targets)
        if not payload or payload.get("_llm_call_failed"):
            return []
        direction, conviction, _, us_dir, eu_dir = decision_from_payload(payload)
        if direction is None:
            return []
        us_tag = payload.get("us_cluster") or (us_dir.name if us_dir else None)
        eu_tag = payload.get("eu_cluster") or (eu_dir.name if eu_dir else None)
        conv = float(payload.get("_conviction_normalized", conviction))
        return resolve_orders_from_cluster_tags(
            bar_index=bar_index,
            snapshot=snapshot,
            config=config,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            all_symbols=all_syms,
            anchor=anchor,
            us_tag=str(us_tag) if us_tag else None,
            eu_tag=str(eu_tag) if eu_tag else None,
            conviction=conv,
            runtime=runtime,
        )

    orders, stats = await collect_hft_with_resolver(
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        all_syms=all_syms,
        anchor=anchor,
        macro_cfg=macro_cfg,
        start=start,
        end=end,
        resolver=_resolver,
        config=config,
    )
    stats.update(
        {
            "gemini_llm_calls": llm_calls,
            "gemini_llm_failures": llm_failures,
            "gemini_cache_path": str(cache_file) if cache_file else "",
            "gemini_bar_step": step,
            "gemini_schedule": schedule,
            "gemini_query_points": len(targets),
        }
    )
    return orders, stats
