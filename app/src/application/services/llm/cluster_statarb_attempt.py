"""Ordem de tentativa de indices StatArb no cluster."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_statarb_score import alignment_score, wr_blend_score
from src.application.services.llm.cluster_statarb_select import select_cluster_symbols_by_statarb
from src.domain.models.trade import TradeDirection


def select_cluster_symbol_attempt_order(
    candidates: list[str],
    direction: TradeDirection,
    statarb_spreads: dict[str, float] | None,
    *,
    hmm_state: int = 0,
    cfg: dict[str, Any] | None = None,
    wr_scores: dict[str, float] | None = None,
) -> tuple[list[str], str, set[str]]:
    """Ordem de tentativa: lideres StatArb primeiro, depois demais candidatos por score."""
    picked, note = select_cluster_symbols_by_statarb(
        candidates,
        direction,
        statarb_spreads,
        hmm_state=hmm_state,
        cfg=cfg,
        wr_scores=wr_scores,
    )
    base_cfg = cfg if isinstance(cfg, dict) else {}
    if base_cfg.get("execute_all") or not base_cfg.get("enabled", True):
        return list(candidates), note, picked
    spreads = statarb_spreads or {}
    wr_weight = float(base_cfg.get("wr_weight", 0.0))
    scored: list[tuple[str, float, float]] = []
    for sym in candidates:
        z = spreads.get(sym)
        if z is None:
            continue
        zf = float(z)
        align = alignment_score(zf, direction, hmm_state)
        composite = align + wr_blend_score(sym, wr_scores, wr_weight)
        scored.append((sym, zf, composite))
    ranked = [row[0] for row in sorted(scored, key=lambda row: (row[2], abs(row[1])), reverse=True)]
    order: list[str] = []
    for sym in ranked:
        if sym in picked and sym not in order:
            order.append(sym)
    for sym in ranked:
        if sym not in order:
            order.append(sym)
    for sym in candidates:
        if sym not in order:
            order.append(sym)
    return order, note, picked
