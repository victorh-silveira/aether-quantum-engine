"""Snapshot macro Medallion em um instante do backtest (sem orchestrator)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.macro_config import resolve_macro_config
from src.application.services.llm.macro_index_m5 import build_index_m5_dir_map
from src.application.services.llm.macro_snapshot_build import apply_m5_fallback_to_snapshot, build_macro_snapshot
from src.application.services.llm.medallion_statarb import (
    KalmanFilter,
    MarketHMMClassifier,
    compute_pca_cointegration_zscores,
)
from src.application.services.llm.synthetic_universe import DEFAULT_ANCHOR


def _window(series: list[float], end_index: int, max_len: int) -> list[float]:
    """Recorta serie ate end_index inclusive com no maximo max_len pontos."""
    if end_index < 0 or not series:
        return []
    chunk = series[: end_index + 1]
    if len(chunk) > max_len:
        return chunk[-max_len:]
    return chunk


def _hmm_from_anchor(anchor_closes: list[float], cfg: dict[str, Any]) -> tuple[int, float]:
    """Replica HMM do marcapasso em macro_snapshot_fetch."""
    if len(anchor_closes) < 3:
        return 0, 1.0
    kf = KalmanFilter(q=1e-5, r=1e-3)
    denoised = kf.filter_series(anchor_closes)
    log_returns = np.diff(np.log(denoised))
    hmm = MarketHMMClassifier(
        sigma_low=float(cfg["statarb_hmm_sigma_low"]),
        sigma_high=float(cfg["statarb_hmm_sigma_high"]),
    )
    state = 0
    prob = 1.0
    for ret in log_returns:
        state, prob = hmm.update_regime(float(ret))
    return state, prob


def build_snapshot_at_bar(
    *,
    bar_index: int,
    m15_closes: dict[str, list[float]],
    m5_closes: dict[str, list[float]],
    us_symbols: list[str],
    eu_symbols: list[str],
    macro_cfg: dict[str, Any] | None,
    anchor: str = DEFAULT_ANCHOR,
) -> MacroSnapshot:
    """Monta MacroSnapshot no fechamento da barra primaria bar_index."""
    cfg = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
    bars = int(cfg.get("cluster_bars", 8))
    statarb_lookback = int(cfg.get("statarb_lookback", 30))
    window_len = max(bars, statarb_lookback)

    closes_map: dict[str, list[float]] = {}
    anchor_sym = str(anchor or DEFAULT_ANCHOR)
    all_syms = list(dict.fromkeys(us_symbols + eu_symbols + [anchor_sym]))
    for sym in all_syms:
        closes_map[sym] = _window(m15_closes.get(sym, []), bar_index, window_len)

    hmm_state, hmm_prob = _hmm_from_anchor(closes_map.get(anchor_sym, []), cfg)
    all_indices = us_symbols + eu_symbols
    statarb_spreads = compute_pca_cointegration_zscores(
        closes_map,
        all_indices,
        lookback=statarb_lookback,
    )

    snap = build_macro_snapshot(
        us_symbols,
        eu_symbols,
        closes_map,
        macro_cfg if isinstance(macro_cfg, dict) else None,
        statarb_spreads=statarb_spreads,
        hmm_state=hmm_state,
        hmm_prob=hmm_prob,
    )

    fb_gran_ratio = 3
    m5_end = min(
        len(m5_closes.get(us_symbols[0] if us_symbols else anchor, [])) - 1,
        (bar_index + 1) * fb_gran_ratio,
    )
    m5_window = max(int(cfg.get("cluster_fallback_bars", 12)), 2)
    m5_slice: dict[str, list[float]] = {}
    for sym in dict.fromkeys(all_indices + [anchor]):
        m5_slice[sym] = _window(m5_closes.get(sym, []), m5_end, m5_window)
    index_m5_dirs = build_index_m5_dir_map(
        m5_slice,
        macro_cfg if isinstance(macro_cfg, dict) else None,
    )
    if index_m5_dirs:
        snap = MacroSnapshot(
            us_dir=snap.us_dir,
            eu_dir=snap.eu_dir,
            us_strength=snap.us_strength,
            eu_strength=snap.eu_strength,
            tag=snap.tag,
            eurusd_bias=snap.eurusd_bias,
            cluster_status=snap.cluster_status,
            macro_block=snap.macro_block,
            fx_reference_line=snap.fx_reference_line,
            us_parts=snap.us_parts,
            eu_parts=snap.eu_parts,
            statarb_spreads=snap.statarb_spreads,
            hmm_state=snap.hmm_state,
            hmm_prob=snap.hmm_prob,
            index_m5_dir_by_symbol=index_m5_dirs,
        )

    if not cfg["cluster_use_m5_fallback_when_flat"]:
        return snap
    if snap.us_dir != "flat" and snap.eu_dir != "flat":
        return snap

    fb_bars = int(cfg.get("cluster_fallback_bars", 12))
    fb_closes: dict[str, list[float]] = {}
    fb_syms: list[str] = []
    if snap.us_dir == "flat":
        fb_syms.extend(us_symbols)
    if snap.eu_dir == "flat":
        fb_syms.extend(eu_symbols)
    for sym in dict.fromkeys(fb_syms):
        fb_closes[sym] = _window(m5_closes.get(sym, []), m5_end, fb_bars)

    return apply_m5_fallback_to_snapshot(
        snap,
        us_symbols=us_symbols,
        eu_symbols=eu_symbols,
        fallback_closes=fb_closes,
        macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
    )
