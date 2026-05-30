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
from src.application.services.llm.synthetic_universe import (
    DEFAULT_ANCHOR,
    default_strategy_clusters,
    resolve_anchor as resolve_config_anchor,
)


def _stream_fetch_is_async(orch: Any) -> bool:
    """True quando fetch_candle_closes do stream e awaitable."""
    fetch = getattr(getattr(orch, "stream", None), "fetch_candle_closes", None)
    if fetch is None:
        return False
    if hasattr(fetch, "mock_calls"):
        return False
    return asyncio.iscoroutinefunction(fetch) or bool(getattr(fetch, "_is_coroutine", False))


async def _fetch_m5_closes(orch: Any, symbols: list[str], gran: int, bars: int) -> dict[str, list[float]]:
    """Busca closes M5 para simbolos em paralelo."""
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


def _resolve_cluster_symbols(strategy: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Resolve listas US/EU a partir de strategy.clusters ou defaults sinteticos."""
    clusters = strategy.get("clusters") if isinstance(strategy.get("clusters"), dict) else None
    if clusters is not None:
        us, eu = resolve_cluster_lists(strategy)
        if us or eu:
            return us, eu
    defaults = default_strategy_clusters()
    return list(defaults["us"]), list(defaults["eu"])


def _hmm_from_anchor(anchor_closes: list[float], cfg: dict[str, Any]) -> tuple[int, float]:
    """Calcula estado e probabilidade HMM no simbolo ancora."""
    if len(anchor_closes) < 3:
        return 0, 1.0
    kf = KalmanFilter(q=1e-5, r=1e-3)
    denoised_eurusd = kf.filter_series(anchor_closes)
    log_returns = np.diff(np.log(denoised_eurusd))
    hmm = MarketHMMClassifier(
        sigma_low=float(cfg["statarb_hmm_sigma_low"]),
        sigma_high=float(cfg["statarb_hmm_sigma_high"]),
    )
    state = 0
    prob = 1.0
    for ret in log_returns:
        state, prob = hmm.update_regime(ret)
    return state, prob


async def _apply_m5_flat_fallback(
    orch: Any,
    snap: MacroSnapshot,
    *,
    us_symbols: list[str],
    eu_symbols: list[str],
    m5_closes: dict[str, list[float]],
    macro_cfg: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> MacroSnapshot:
    """Aplica fallback M5 quando votos M15 do cluster estao flat."""
    fb_syms: list[str] = []
    if snap.us_dir == "flat":
        fb_syms.extend(us_symbols)
    if snap.eu_dir == "flat":
        fb_syms.extend(eu_symbols)
    fb_unique = list(dict.fromkeys(fb_syms))
    fb_closes = dict(m5_closes)
    extra_syms = [s for s in fb_unique if s not in fb_closes]
    fb_gran = int(cfg["cluster_fallback_granularity_seconds"])
    fb_bars = int(cfg["cluster_fallback_bars"])
    extra = await _fetch_m5_closes(orch, extra_syms, fb_gran, fb_bars)
    fb_closes.update(extra)
    snap = apply_m5_fallback_to_snapshot(
        snap,
        us_symbols=us_symbols,
        eu_symbols=eu_symbols,
        fallback_closes=fb_closes,
        macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
    )
    return replace(
        snap,
        index_m5_dir_by_symbol=build_index_m5_dir_map(
            fb_closes,
            macro_cfg if isinstance(macro_cfg, dict) else None,
        ),
    )


async def fetch_macro_snapshot(orch: Any, runtime: dict[str, Any]) -> MacroSnapshot:
    """Monta snapshot macro com StatArb, HMM e direcoes M5 por indice."""
    if not _stream_fetch_is_async(orch):
        return empty_macro_snapshot()

    try:
        strategy = orch.config.get("strategy", {}) if hasattr(orch, "config") and isinstance(orch.config, dict) else {}
        us_symbols, eu_symbols = _resolve_cluster_symbols(strategy if isinstance(strategy, dict) else {})
        macro_cfg = strategy.get("macro")
        cfg = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
        swing_gran = int(cfg.get("cluster_granularity_seconds", runtime.get("tf_swing_gran", 900)))
        bars = int(cfg.get("cluster_bars", 8))
        statarb_lookback = int(cfg.get("statarb_lookback", 15))
        bars_to_fetch = max(bars, statarb_lookback)
        fb_gran = int(cfg["cluster_fallback_granularity_seconds"])
        fb_bars = int(cfg["cluster_fallback_bars"])

        anchor = DEFAULT_ANCHOR
        if hasattr(orch, "anchor") and orch.anchor:
            anchor = str(orch.anchor)
        elif hasattr(orch, "config") and isinstance(orch.config, dict):
            anchor = resolve_config_anchor(orch.config)
        all_syms = list(dict.fromkeys(us_symbols + eu_symbols + [anchor]))
        results = await asyncio.gather(
            *[orch.stream.fetch_candle_closes(s, swing_gran, bars_to_fetch) for s in all_syms],
            return_exceptions=True,
        )
        closes_map: dict[str, list[float]] = {}
        for i, sym in enumerate(all_syms):
            row = results[i]
            closes_map[sym] = list(row) if isinstance(row, list) else []

        hmm_state, hmm_prob = _hmm_from_anchor(closes_map.get(anchor, []), cfg)
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
            snap = await _apply_m5_flat_fallback(
                orch,
                snap,
                us_symbols=us_symbols,
                eu_symbols=eu_symbols,
                m5_closes=m5_closes,
                macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
                cfg=cfg,
            )
        return snap
    except Exception as e:
        if hasattr(orch, "logger") and orch.logger:
            orch.logger.warning("Error fetching macro snapshot: %s", e)
        return empty_macro_snapshot()
