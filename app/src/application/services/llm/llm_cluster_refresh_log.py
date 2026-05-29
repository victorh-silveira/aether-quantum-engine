"""Helpers de log para refresh de cluster com direcao efetiva."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def _dir_label(entry: dict[str, Any] | None) -> str:
    """Formata direcao executavel de uma entrada de decisao para log."""
    if not isinstance(entry, dict):
        return "-"
    direction = entry.get("direction")
    if isinstance(direction, TradeDirection):
        return direction.name
    return "-"


def effective_cluster_refresh_line(
    decisions: dict[str, dict],
    *,
    anchor_sym: str,
    metrics: dict[str, Any],
    macro_tag: str,
) -> str:
    """Monta linha CLUSTER_REFRESH com direcao efetiva e cache LLM."""
    us_eff = "-"
    eu_eff = "-"
    for sym, entry in decisions.items():
        if sym == anchor_sym:
            continue
        if not isinstance(entry, dict):
            continue
        m = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        region = str(m.get("cluster_active_region") or "")
        label = _dir_label(entry)
        if region == "us":
            us_eff = label
        elif region == "eu":
            eu_eff = label
    llm_us = str(metrics.get("us_cluster") or "-")
    llm_eu = str(metrics.get("eu_cluster") or "-")
    return f"macro={macro_tag} | us_eff={us_eff} eu_eff={eu_eff} | llm_cache={llm_us}/{llm_eu}"
