"""Pisos de qualidade de sinal para execucao em modo recovery."""

from __future__ import annotations

from src.application.services.execution_direction import mandatory_execution_eligible, recovery_execution_eligible


def recovery_min_signal(kelly_config: dict, *, recovery_active: bool) -> float:
    """Piso de trade_score para pool e fallback obrigatorio."""
    floor = float(kelly_config.get("mandatory_min_trade_score", 0.45))
    if not recovery_active:
        return floor
    recovery_floor = float(kelly_config.get("recovery_min_trade_score", floor))
    return max(floor, recovery_floor)


def recovery_min_val_accuracy(kelly_config: dict) -> float:
    """Piso de val_accuracy para candidatos de recovery obrigatorio."""
    return float(kelly_config.get("recovery_min_val_accuracy", 0.50))


def cluster_entry_eligible(
    entry: dict,
    *,
    mandatory: bool,
    recovery_active: bool,
    recovery_cfg: dict,
    min_signal: float,
    min_val: float,
) -> bool:
    """Indica se entrada DL pode entrar no pool de candidatos do ciclo."""
    may_execute = bool(entry.get("metrics", {}).get("execute", False))
    if may_execute:
        return True
    if recovery_active:
        return recovery_execution_eligible(entry, recovery_cfg)
    return mandatory and mandatory_execution_eligible(entry, min_signal=min_signal, min_val_accuracy=min_val)
