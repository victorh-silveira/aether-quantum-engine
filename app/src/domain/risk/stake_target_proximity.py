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
