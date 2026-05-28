"""Fallbacks StatArb quando o filtro estrito de Z falha no cluster."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.llm_macro_confluence_guards import _statarb_misaligned
from src.domain.models.trade import TradeDirection


def _statarb_soft_fallback_pick(
    ranked: list[tuple[str, float, float]],
    direction: TradeDirection,
    *,
    hmm_state: int,
    z_threshold: float,
    min_abs: float,
    max_per_cluster: int,
    soft_min_abs_ratio: float = 0.45,
) -> tuple[set[str], str] | None:
    """Escolhe lider via soft fallback quando o filtro estrito de Z falha."""
    soft_rows: list[tuple[str, float, float]] = []
    soft_floor = min_abs * max(0.1, float(soft_min_abs_ratio))
    for row in ranked:
        zf = row[1]
        if _statarb_misaligned(direction, zf, z_threshold, hmm_state):
            continue
        if row[2] > 0.0 or abs(zf) >= soft_floor:
            soft_rows.append(row)
    if not soft_rows:
        return None
    leader_rows = soft_rows[:max_per_cluster]
    picked = {row[0] for row in leader_rows}
    leader = leader_rows[0]
    note = f"STATARB_SOFT leader={leader[0]} z={leader[1]:.2f} score={leader[2]:.2f} n={len(picked)}"
    return picked, note


def _statarb_weak_leader_pick(
    ranked: list[tuple[str, float, float]],
    direction: TradeDirection,
    *,
    hmm_state: int,
    z_threshold: float,
    max_per_cluster: int,
) -> tuple[set[str], str] | None:
    """Ultimo recurso: melhor alinhamento sem piso de |Z| quando estrito e soft falham."""
    weak_rows: list[tuple[str, float, float]] = []
    for row in ranked:
        zf = row[1]
        if _statarb_misaligned(direction, zf, z_threshold, hmm_state):
            continue
        if row[2] > 0.0:
            weak_rows.append(row)
    if not weak_rows:
        return None
    leader_rows = weak_rows[:max_per_cluster]
    picked = {row[0] for row in leader_rows}
    leader = leader_rows[0]
    note = f"STATARB_WEAK leader={leader[0]} z={leader[1]:.2f} score={leader[2]:.2f} n={len(picked)}"
    return picked, note


def statarb_relaxed_pick(
    ranked: list[tuple[str, float, float]],
    direction: TradeDirection,
    *,
    hmm_state: int,
    z_threshold: float,
    min_abs: float,
    max_per_cluster: int,
    base_cfg: dict[str, Any],
) -> tuple[set[str], str] | None:
    """Encadeia soft fallback e weak leader quando o filtro estrito de Z falha."""
    if bool(base_cfg.get("z_align_soft_fallback", False)) and ranked:
        soft_pick = _statarb_soft_fallback_pick(
            ranked,
            direction,
            hmm_state=hmm_state,
            z_threshold=z_threshold,
            min_abs=min_abs,
            max_per_cluster=max_per_cluster,
            soft_min_abs_ratio=float(base_cfg.get("soft_min_abs_ratio", 0.45)),
        )
        if soft_pick is not None:
            return soft_pick
    if bool(base_cfg.get("weak_leader_on_no_align", True)) and ranked:
        return _statarb_weak_leader_pick(
            ranked,
            direction,
            hmm_state=hmm_state,
            z_threshold=z_threshold,
            max_per_cluster=max_per_cluster,
        )
    return None
