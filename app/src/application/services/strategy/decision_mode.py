"""Resolucao do modo de decisao ativo (Deep Learning)."""

from __future__ import annotations

from typing import Any


def resolve_decision_mode(config: dict[str, Any]) -> str:
    """Prioriza setup operacional e mantem DL apenas no modo legado."""
    if bool((config.get("pullback_strategy") or {}).get("enabled")):
        return "pullback"
    if bool((config.get("deep_learning") or {}).get("enabled")):
        return "deep_learning"
    return "inactive"
