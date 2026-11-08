"""Decaimento temporal do piso Hurst em recovery por ciclos de SKIP consecutivos."""

from __future__ import annotations

import math
from typing import Any


REDIS_SKIP_COUNTER_KEY = "recovery:skip_counter"


def session_drawdown_from_profit(total_session_profit: float) -> float:
    """Converte PnL de sessao em drawdown positivo (USD perdidos)."""
    return max(0.0, -float(total_session_profit))


def effective_recovery_hurst_min(
    base_min: float,
    skip_counter: int,
    *,
    decay: float = 0.01,
    floor: float = 0.50,
    consecutive_losses: int = 0,
    session_drawdown: float = 0.0,
    log_decay_coef: float = 0.025,
    losses_accel_min: int = 3,
    severe_drawdown_min: float = 0.0,
) -> float:
    """Reduz o limiar Hurst linearmente ou com decaimento log acelerado em drawdown severo."""
    base = float(base_min)
    count = max(0, int(skip_counter))
    bound = float(floor)
    losses = int(consecutive_losses)
    drawdown = max(0.0, float(session_drawdown))
    if losses >= int(losses_accel_min) and drawdown + 1e-9 >= float(severe_drawdown_min):
        reduction = float(log_decay_coef) * math.log1p(count)
        return max(bound, base - reduction)
    step = max(0.0, float(decay))
    return max(bound, base - step * count)


def resolve_effective_hurst_min(
    kelly_cfg: dict[str, Any],
    skip_counter: int,
    *,
    consecutive_losses: int = 0,
    session_drawdown: float = 0.0,
) -> float:
    """Aplica decay configuravel quando habilitado."""
    cfg = kelly_cfg if isinstance(kelly_cfg, dict) else {}
    base = float(cfg.get("recovery_hurst_persistence_min", 0.58))
    if not bool(cfg.get("recovery_hurst_decay_enabled", True)):
        return base
    return effective_recovery_hurst_min(
        base,
        skip_counter,
        decay=float(cfg.get("recovery_hurst_decay_per_skip", 0.01)),
        floor=float(cfg.get("recovery_hurst_decay_floor", 0.50)),
        consecutive_losses=consecutive_losses,
        session_drawdown=session_drawdown,
        log_decay_coef=float(cfg.get("recovery_hurst_log_decay_coef", 0.025)),
        losses_accel_min=int(cfg.get("recovery_hurst_accel_losses_min", 3)),
        severe_drawdown_min=float(cfg.get("recovery_hurst_severe_drawdown_min", 150.0)),
    )
