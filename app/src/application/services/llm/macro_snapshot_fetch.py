"""Busca de fechamentos e montagem de snapshot macro com fallback M5."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import numpy as np

from src.application.services.llm.global_macro_confluence import (
    MacroSnapshot,
    apply_m5_fallback_to_snapshot,
    build_macro_snapshot,
    empty_macro_snapshot,
    resolve_macro_config,
)
from src.application.services.llm.macro_index_m5 import build_index_m5_dir_map
from src.application.services.llm.medallion_statarb import (
    KalmanFilter,
    MarketHMMClassifier,
    compute_pca_cointegration_zscores,
)
from src.application.services.llm.strategy_clusters import resolve_cluster_lists


def _stream_fetch_is_async(orch: Any) -> bool:
    fetch = getattr(getattr(orch, "stream", None), "fetch_candle_closes", None)
    if fetch is None:
        return False
    if hasattr(fetch, "mock_calls"):
        return False
    return asyncio.iscoroutinefunction(fetch) or bool(getattr(fetch, "_is_coroutine", False))


async def _fetch_m5_closes(orch: Any, symbols: list[str], gran: int, bars: int) -> dict[str, list[float]]:
    if not symbols:
        return {}
    results = await asyncio.gather(
        *[orch.stream.fetch_candle_closes(s, gran, bars) for s in symbols],
        return_exceptions=True,
    )
    out: dict[str, list[float]] = {}
    for i, sym in enumerate(symbols):
        row = results[i]
        out[sym] = list(row) if isinstance(row, list) else []
    return out


async def fetch_macro_snapshot(orch: Any, runtime: dict[str, Any]) -> MacroSnapshot:
    if not _stream_fetch_is_async(orch):
        return empty_macro_snapshot()

    try:
        strategy = orch.config.get("strategy", {}) if hasattr(orch, "config") and isinstance(orch.config, dict) else {}
        clusters = strategy.get("clusters") if isinstance(strategy.get("clusters"), dict) else None
        if clusters is not None:
            us_symbols, eu_symbols = resolve_cluster_lists(strategy if isinstance(strategy, dict) else None)
        else:
            us_symbols = ["OTC_NDX", "OTC_DJI"]
            eu_symbols = ["OTC_FCHI", "OTC_GDAXI", "OTC_FTSE"]
        macro_cfg = strategy.get("macro")
        cfg = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
        swing_gran = int(cfg.get("cluster_granularity_seconds", runtime.get("tf_swing_gran", 900)))
        bars = int(cfg.get("cluster_bars", 8))
        statarb_lookback = int(cfg.get("statarb_lookback", 15))
        bars_to_fetch = max(bars, statarb_lookback)
        fb_gran = int(cfg["cluster_fallback_granularity_seconds"])
        fb_bars = int(cfg["cluster_fallback_bars"])

        all_syms = us_symbols + eu_symbols + ["frxEURUSD"]
        results = await asyncio.gather(
            *[orch.stream.fetch_candle_closes(s, swing_gran, bars_to_fetch) for s in all_syms],
            return_exceptions=True,
        )
        closes_map: dict[str, list[float]] = {}
        for i, sym in enumerate(all_syms):
            row = results[i]
            closes_map[sym] = list(row) if isinstance(row, list) else []

        eurusd_closes = closes_map.get("frxEURUSD", [])
        hmm_state = 0
        hmm_prob = 1.0
        if len(eurusd_closes) >= 3:
            kf = KalmanFilter(q=1e-5, r=1e-3)
            denoised_eurusd = kf.filter_series(eurusd_closes)
            log_returns = np.diff(np.log(denoised_eurusd))
            hmm = MarketHMMClassifier(
                sigma_low=float(cfg["statarb_hmm_sigma_low"]),
                sigma_high=float(cfg["statarb_hmm_sigma_high"]),
            )
            for ret in log_returns:
                hmm_state, hmm_prob = hmm.update_regime(ret)

        all_indices = us_symbols + eu_symbols
        statarb_spreads = compute_pca_cointegration_zscores(
            closes_map,
            all_indices,
            lookback=statarb_lookback,
        )

        m5_syms = list(dict.fromkeys(us_symbols + eu_symbols))
        m5_closes = await _fetch_m5_closes(orch, m5_syms, fb_gran, fb_bars)
        index_m5_dirs = build_index_m5_dir_map(m5_closes, macro_cfg if isinstance(macro_cfg, dict) else None)

        snap = build_macro_snapshot(
            us_symbols,
            eu_symbols,
            closes_map,
            macro_cfg,
            statarb_spreads=statarb_spreads,
            hmm_state=hmm_state,
            hmm_prob=hmm_prob,
            index_m5_dir_by_symbol=index_m5_dirs,
        )

        if (snap.us_dir == "flat" or snap.eu_dir == "flat") and cfg["cluster_use_m5_fallback_when_flat"]:
            fb_syms: list[str] = []
            if snap.us_dir == "flat":
                fb_syms.extend(us_symbols)
            if snap.eu_dir == "flat":
                fb_syms.extend(eu_symbols)
            fb_unique = list(dict.fromkeys(fb_syms))
            fb_closes = dict(m5_closes)
            extra = await _fetch_m5_closes(orch, [s for s in fb_unique if s not in fb_closes], fb_gran, fb_bars)
            fb_closes.update(extra)
            snap = apply_m5_fallback_to_snapshot(
                snap,
                us_symbols=us_symbols,
                eu_symbols=eu_symbols,
                fallback_closes=fb_closes,
                macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
            )
            snap = replace(
                snap,
                index_m5_dir_by_symbol=build_index_m5_dir_map(
                    fb_closes,
                    macro_cfg if isinstance(macro_cfg, dict) else None,
                ),
            )
        return snap
    except Exception as e:
        if hasattr(orch, "logger") and orch.logger:
            orch.logger.warning("Error fetching macro snapshot: %s", e)
        return empty_macro_snapshot()
