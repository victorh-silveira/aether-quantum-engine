"""Tipos e configuracao para confluencia macro transatlantica."""

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


def resolve_macro_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza configuracao strategy.macro com defaults seguros."""
    base = raw if isinstance(raw, dict) else {}
    fx_raw = base.get("fx_reference_pairs")
    fx_pairs = fx_raw if isinstance(fx_raw, dict) else {}
    labels_raw = base.get("cluster_labels")
    labels = _normalize_cluster_labels(labels_raw if isinstance(labels_raw, dict) else {})
    return {
        "cluster_return_threshold_pct": float(base.get("cluster_return_threshold_pct", 0.02)),
        "min_indices_for_vote": max(1, int(base.get("min_indices_for_vote", 2))),
        "divergence_blocks_execution": bool(base.get("divergence_blocks_execution", True)),
        "divergence_max_conviction": max(0.0, min(0.99, float(base.get("divergence_max_conviction", 0.65)))),
        "align_eurusd_with_confluence": bool(base.get("align_eurusd_with_confluence", True)),
        "align_clusters_with_macro_vote": bool(base.get("align_clusters_with_macro_vote", True)),
        "confluence_conviction_floor": max(0.0, min(0.99, float(base.get("confluence_conviction_floor", 0.55)))),
        "fx_reference_pairs": fx_pairs,
        "cluster_labels": labels,
    }


def _normalize_cluster_labels(raw: dict[str, Any]) -> dict[str, list[str]]:
    """Normaliza rotulos de cluster para maiusculas."""
    out: dict[str, list[str]] = {}
    for region, items in raw.items():
        if isinstance(items, list):
            out[str(region)] = [str(x).upper() for x in items]
    return out
