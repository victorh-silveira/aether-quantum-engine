"""Piso logaritmico de recovery por Hurst e filtro de persistencia do pool."""

from __future__ import annotations

import math
from typing import Any

from src.domain.risk.recovery_hurst_decay import resolve_effective_hurst_min


def recovery_loss_tier_floor(base: float, consecutive_losses: int) -> float:
    """Eleva piso base conforme streak de perdas consecutivas."""
    losses = int(consecutive_losses)
    if losses == 1:
        return max(base, 0.52)
    if losses == 2:
        return max(base, 0.54)
    if losses == 3:
        return max(base, 0.56)
    if losses >= 4:
        return max(base, 0.58)
    return base


def resolve_recovery_signal_floor(
    kelly_config: dict[str, Any],
    *,
    hurst: float,
    consecutive_losses: int,
    total_session_profit: float,
    recovery_skip_counter: int = 0,
) -> float:
    """Piso de sinal em recovery ajustado por streak, Hurst efetivo e decaimento Redis."""
    base = float(kelly_config.get("recovery_min_trade_score", 0.64))
    base = recovery_loss_tier_floor(base, consecutive_losses)
    losses = int(consecutive_losses)
    hurst_min = resolve_effective_hurst_min(
        kelly_config,
        recovery_skip_counter,
        consecutive_losses=losses,
        session_drawdown=max(0.0, -float(total_session_profit)),
    )
    return recovery_hurst_adjusted_floor(
        base,
        float(hurst),
        consecutive_losses=losses,
        hurst_persistence_min=hurst_min,
        log_scale=float(kelly_config.get("recovery_hurst_log_scale", 0.08)),
    )


def recovery_hurst_adjusted_floor(
    base_floor: float,
    hurst: float,
    *,
    consecutive_losses: int,
    hurst_persistence_min: float = 0.58,
    log_scale: float = 0.08,
) -> float:
    """Eleva piso de sinal quando Hurst nao indica persistencia em martingale N2+."""
    if int(consecutive_losses) < 2:
        return float(base_floor)
    h = float(hurst)
    target = float(hurst_persistence_min)
    if h + 1e-9 >= target:
        return float(base_floor)
    deficit = max(0.0, target - h)
    return float(base_floor) + float(log_scale) * math.log1p(deficit)


def recovery_pool_has_persistence(
    candidates: list[tuple],
    *,
    consecutive_losses: int,
    hurst_min: float = 0.58,
) -> bool:
    """True quando ao menos um candidato tem Hurst acima do minimo de persistencia."""
    if int(consecutive_losses) < 2:
        return True
    threshold = float(hurst_min)
    for item in candidates:
        if not isinstance(item, tuple) or len(item) < 3:
            continue
        metrics = item[2]
        if not isinstance(metrics, dict):
            continue
        indicators = metrics.get("indicators") or {}
        if float(indicators.get("hurst", 0.0)) + 1e-9 > threshold:
            return True
    return False
