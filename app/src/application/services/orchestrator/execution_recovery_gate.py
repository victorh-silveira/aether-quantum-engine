"""Pisos de qualidade de sinal para execucao em modo recovery."""

from __future__ import annotations

from src.application.services.execution_direction_resolver import is_technically_blocked
from src.domain.risk.recovery_hurst_gate import recovery_hurst_adjusted_floor
from src.domain.risk.stake_sizing import raw_side_from_metrics


def recovery_min_signal(
    kelly_config: dict,
    *,
    recovery_active: bool,
    pending_total: float = 0.0,
    consecutive_losses: int = 0,
    hurst: float | None = None,
    hurst_persistence_min: float | None = None,
) -> float:
    """Piso de trade_score para recovery linear e sizing em recovery."""
    floor = float(kelly_config.get("mandatory_min_trade_score", 0.45))
    if not recovery_active:
        return floor
    recovery_floor = float(kelly_config.get("recovery_min_trade_score", floor))
    force_min = float(kelly_config.get("recovery_force_min_trade_score", recovery_floor))
    force_pending = float(kelly_config.get("recovery_force_pending_min", 0.0))
    sig_floor = recovery_floor
    if force_pending > 0.0 and float(pending_total) + 1e-9 >= force_pending:
        sig_floor = min(recovery_floor, force_min)

    losses = int(consecutive_losses)
    if losses == 1:
        sig_floor = max(sig_floor, 0.52)
    elif losses == 2:
        sig_floor = max(sig_floor, 0.54)
    elif losses == 3:
        sig_floor = max(sig_floor, 0.56)
    elif losses >= 4:
        sig_floor = max(sig_floor, 0.58)

    if hurst is not None:
        persistence_min = (
            float(hurst_persistence_min)
            if hurst_persistence_min is not None
            else float(kelly_config.get("recovery_hurst_persistence_min", 0.58))
        )
        sig_floor = recovery_hurst_adjusted_floor(
            sig_floor,
            float(hurst),
            consecutive_losses=losses,
            hurst_persistence_min=persistence_min,
            log_scale=float(kelly_config.get("recovery_hurst_log_scale", 0.08)),
        )
    return sig_floor


def recovery_min_val_accuracy(
    kelly_config: dict,
    *,
    consecutive_losses: int = 0,
) -> float:
    """Piso de val_accuracy para recovery linear em recovery."""
    base_val = float(kelly_config.get("recovery_min_val_accuracy", 0.50))

    losses = int(consecutive_losses)
    if losses == 2:
        base_val = max(base_val, 0.52)
    elif losses == 3:
        base_val = max(base_val, 0.53)
    elif losses >= 4:
        base_val = max(base_val, 0.55)

    return base_val


def cluster_entry_eligible(
    entry: dict,
    *,
    mandatory: bool,  # noqa: ARG001
    recovery_active: bool,  # noqa: ARG001
    recovery_cfg: dict,  # noqa: ARG001
    min_signal: float = 0.0,  # noqa: ARG001
    min_val: float = 0.0,  # noqa: ARG001
    min_edge: float = 0.0,  # noqa: ARG001
) -> bool:
    """Indica se entrada DL pode entrar no pool; bloqueia apenas falhas tecnicas."""
    if is_technically_blocked(entry):
        return False
    metrics = entry.get("metrics") or {}
    return metrics.get("raw_prob") is not None or entry.get("direction") is not None


def effective_signal(metrics: dict) -> float:
    """Retorna o maior entre trade_score calibrado e conviccao bruta lateralizada."""
    score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    return max(score, raw_side_from_metrics(metrics))
