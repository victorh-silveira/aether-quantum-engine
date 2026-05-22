"""Selecao Medallion de indices dentro do cluster ativo via Z-Score StatArb."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.macro_config import resolve_macro_config
from src.domain.models.trade import TradeDirection


def resolve_statarb_cluster_config(corr: dict[str, Any] | None, macro: dict[str, Any] | None) -> dict[str, Any]:
    """Mescla flags de selecao por indice em correlation e strategy.macro."""
    c = corr if isinstance(corr, dict) else {}
    m = resolve_macro_config(macro if isinstance(macro, dict) else None)
    enabled = bool(c.get("statarb_index_select_enabled", c.get("statarb_index_select", True)))
    return {
        "enabled": enabled,
        "max_per_cluster": max(1, int(c.get("statarb_index_max_per_cluster", c.get("statarb_index_max", 1)))),
        "min_abs_z": max(0.0, float(c.get("statarb_index_min_abs_z", 0.0))),
        "z_threshold": float(m["statarb_z_threshold"]),
    }


def _alignment_score(z: float, direction: TradeDirection, hmm_state: int) -> float:
    """Pontua alinhamento Z com direcao do cluster; HMM tendencia reduz peso."""
    if direction == TradeDirection.CALL:
        raw = max(0.0, -z)
    elif direction == TradeDirection.PUT:
        raw = max(0.0, z)
    else:
        return 0.0
    if hmm_state == 1:
        return raw * 0.5
    return raw


def select_cluster_symbols_by_statarb(
    candidates: list[str],
    direction: TradeDirection,
    statarb_spreads: dict[str, float] | None,
    *,
    hmm_state: int = 0,
    cfg: dict[str, Any] | None = None,
) -> tuple[set[str], str]:
    """Retorna subconjunto de indices com melhor alinhamento StatArb ao cluster."""
    base_cfg = cfg if isinstance(cfg, dict) else {}
    if not base_cfg.get("enabled", True):
        return set(candidates), "STATARB_INDEX_ALL"

    spreads = statarb_spreads or {}
    if not candidates:
        return set(), "STATARB_INDEX_EMPTY"

    scored: list[tuple[str, float, float]] = []
    for sym in candidates:
        z = spreads.get(sym)
        if z is None:
            continue
        zf = float(z)
        scored.append((sym, zf, _alignment_score(zf, direction, hmm_state)))

    if not scored:
        return set(candidates), "STATARB_INDEX_NO_Z_FALLBACK"

    min_abs = float(base_cfg.get("min_abs_z", 0.0))
    ranked = sorted(scored, key=lambda row: (row[2], abs(row[1])), reverse=True)
    filtered = [row for row in ranked if row[2] > 0.0 or abs(row[1]) >= min_abs]
    if not filtered and min_abs > 0.0:
        filtered = ranked[:1]

    max_n = max(1, int(base_cfg.get("max_per_cluster", 1)))
    leader_rows = filtered[:max_n] if filtered else ranked[:1]
    picked = {row[0] for row in leader_rows}
    leader = leader_rows[0]
    note = f"STATARB_INDEX leader={leader[0]} z={leader[1]:.2f} score={leader[2]:.2f} n={len(picked)}"
    return picked, note
