"""Resolucao do modo de decisao ativo (Deep Learning)."""

from __future__ import annotations

from typing import Any


def resolve_decision_mode(config: dict[str, Any]) -> str:
    """Retorna deep_learning ou inactive conforme settings."""
    if bool((config.get("deep_learning") or {}).get("enabled")):
        return "deep_learning"
    return "inactive"
