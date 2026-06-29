"""Hints de regime indicador para o resolver direcional."""

from __future__ import annotations


def indicator_regime_side(metrics: dict) -> None:
    """Define indicator_regime_side em metrics a partir do CMO."""
    indicators = metrics.get("indicators") or {}
    cmo = float(indicators.get("cmo", 0.0))
    if cmo > 0.08:
        metrics["indicator_regime_side"] = "call"
    elif cmo < -0.08:
        metrics["indicator_regime_side"] = "put"
