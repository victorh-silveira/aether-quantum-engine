"""Restore de campos do RiskManager a partir de snapshot."""

from typing import Any


def _apply_float_fields(manager: Any, data: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Restaura campos numericos de ponto flutuante no RiskManager."""
    for key in keys:
        if key in data:
            setattr(manager, key, float(data[key]))


def _apply_int_fields(manager: Any, data: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Restaura campos inteiros no RiskManager."""
    for key in keys:
        if key in data:
            setattr(manager, key, int(data[key]))


def apply_risk_snapshot(manager: Any, data: dict[str, Any]) -> None:
    """Aplica dict de risco persistido ao RiskManager."""
    if not isinstance(data, dict) or not data:
        return
    _apply_float_fields(
        manager,
        data,
        ("initial_bankroll", "total_session_profit", "last_martingale_stake", "last_loss_stake"),
    )
    _apply_int_fields(manager, data, ("last_result_tick", "consecutive_losses", "current_cooldown_ticks"))
    rolling = data.get("rolling_wins")
    if isinstance(rolling, dict):
        manager._rolling_wins = {str(k): [int(x) for x in v] for k, v in rolling.items() if isinstance(v, list)}
    pending = data.get("pending_loss")
    if isinstance(pending, dict):
        manager.pending_loss = {str(k): float(v) for k, v in pending.items()}
    streak = data.get("recovery_symbol_loss_streak")
    if isinstance(streak, dict):
        manager.recovery_symbol_loss_streak = {str(k): int(v) for k, v in streak.items()}
    if data.get("last_loss_symbol") is not None:
        manager.last_loss_symbol = str(data["last_loss_symbol"])
    if data.get("last_loss_direction") is not None:
        manager.last_loss_direction = str(data["last_loss_direction"])
