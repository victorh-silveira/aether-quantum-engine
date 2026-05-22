"""Tipos e configuracao para confluencia macro transatlantica Medallion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterVote:
    """Resultado agregado de direcao para um cluster de indices."""

    direction: str
    strength: float
    parts: tuple[str, ...]


@dataclass(frozen=True)
class MacroSnapshot:
    """Snapshot quantitativo de sentimento macro para prompt e guardrails."""

    us_dir: str
    eu_dir: str
    us_strength: float
    eu_strength: float
    tag: str
    eurusd_bias: str
    cluster_status: str
    macro_block: str
    fx_reference_line: str
    us_parts: tuple[str, ...]
    eu_parts: tuple[str, ...]
    statarb_spreads: dict[str, float] = None
    hmm_state: int = 0
    hmm_prob: float = 1.0


def resolve_macro_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza configuracao strategy.macro com defaults Medallion."""
    base = raw if isinstance(raw, dict) else {}
    fx_raw = base.get("fx_reference_pairs")
    fx_pairs = fx_raw if isinstance(fx_raw, dict) else {}
    labels_raw = base.get("cluster_labels")
    labels = _normalize_cluster_labels(labels_raw if isinstance(labels_raw, dict) else {})
    return {
        "cluster_return_threshold_pct": float(base.get("cluster_return_threshold_pct", 0.02)),
        "min_indices_for_vote": max(1, int(base.get("min_indices_for_vote", 2))),
        "divergence_max_conviction": max(0.0, min(0.99, float(base.get("divergence_max_conviction", 0.99)))),
        "confluence_conviction_floor": max(0.0, min(0.99, float(base.get("confluence_conviction_floor", 0.55)))),
        "cluster_min_move_pct": max(0.0, float(base.get("cluster_min_move_pct", 0.06))),
        "cluster_granularity_seconds": max(60, int(base.get("cluster_granularity_seconds", 900))),
        "cluster_bars": max(2, int(base.get("cluster_bars", 8))),
        "cluster_use_m5_fallback_when_flat": bool(base.get("cluster_use_m5_fallback_when_flat", True)),
        "cluster_fallback_granularity_seconds": max(60, int(base.get("cluster_fallback_granularity_seconds", 300))),
        "cluster_fallback_bars": max(2, int(base.get("cluster_fallback_bars", 12))),
        "cluster_fallback_min_move_pct": max(0.0, float(base.get("cluster_fallback_min_move_pct", 0.05))),
        "fx_reference_pairs": fx_pairs,
        "cluster_labels": labels,
        "statarb_z_threshold": float(base.get("statarb_z_threshold", 2.5)),
        "statarb_lookback": int(base.get("statarb_lookback", 15)),
        "statarb_hmm_sigma_low": float(base.get("statarb_hmm_sigma_low", 0.0004)),
        "statarb_hmm_sigma_high": float(base.get("statarb_hmm_sigma_high", 0.0016)),
    }


def _normalize_cluster_labels(raw: dict[str, Any]) -> dict[str, list[str]]:
    """Normaliza rotulos de cluster para maiusculas."""
    out: dict[str, list[str]] = {}
    for region, items in raw.items():
        if isinstance(items, list):
            out[str(region)] = [str(x).upper() for x in items]
    return out
