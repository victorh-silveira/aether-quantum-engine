"""Pisos de qualidade de sinal para execucao em modo recovery."""

from __future__ import annotations


def recovery_min_signal(kelly_config: dict, *, recovery_active: bool) -> float:
    """Piso de trade_score para pool e fallback obrigatorio."""
    _ = recovery_active
    return float(kelly_config.get("mandatory_min_trade_score", 0.45))


def recovery_min_val_accuracy(kelly_config: dict) -> float:
    """Piso de val_accuracy para candidatos de recovery obrigatorio."""
    return float(kelly_config.get("recovery_min_val_accuracy", 0.50))
