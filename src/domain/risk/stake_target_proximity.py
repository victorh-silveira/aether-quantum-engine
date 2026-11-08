"""Amortecimento dinamico de Kelly por proximidade da meta de stop win."""

from typing import Any

from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


def resolve_target_proximity_damping(
    target_win: float,
    session_pnl: float,
    *,
    kelly_config: dict[str, Any] | None = None,
) -> float:
    """Resolve ou aplica resolve target proximity damping."""
    runtime = load_kelly_runtime_from_settings()
    if isinstance(kelly_config, dict):
        floor = (
            float(kelly_config["target_damping_floor"])
            if "target_damping_floor" in kelly_config
            else float(runtime["target_damping_floor"])
        )
        span = (
            float(kelly_config["target_damping_span"])
            if "target_damping_span" in kelly_config
            else float(runtime["target_damping_span"])
        )
    else:
        floor = float(runtime["target_damping_floor"])
        span = float(runtime["target_damping_span"])
    if target_win <= 0.0:
        return 1.0
    remaining_target_pct = max(0.0, (float(target_win) - float(session_pnl)) / float(target_win))
    return floor + span * remaining_target_pct


def apply_target_proximity_damping(
    kelly_stake_raw: float,
    target_win: float,
    session_pnl: float,
    *,
    kelly_config: dict[str, Any] | None = None,
) -> float:
    """Resolve ou aplica apply target proximity damping."""
    damping = resolve_target_proximity_damping(target_win, session_pnl, kelly_config=kelly_config)
    return max(0.0, float(kelly_stake_raw) * damping)


def apply_session_profit_lock(
    stake_raw: float,
    target_win: float,
    session_pnl: float,
    peak_session_profit: float = 0.0,
    *,
    stake_min: float = 1.0,
    lock_fraction: float = 0.25,
    trigger_fraction: float = 0.50,
) -> float:
    """Limita a stake se a sessao ja atingiu >= 50% da meta e recuou, garantindo 25% de lucro."""
    target = float(target_win)
    if target <= 0.0:
        return float(stake_raw)
    current_pnl = float(session_pnl)
    peak = max(float(peak_session_profit), current_pnl)
    if peak + 1e-9 < target * float(trigger_fraction):
        return float(stake_raw)
    locked_profit = target * float(lock_fraction)
    if current_pnl > locked_profit:
        max_allowed_risk = max(float(stake_min), current_pnl - locked_profit)
        return min(float(stake_raw), max_allowed_risk)
    return min(float(stake_raw), float(stake_min))
