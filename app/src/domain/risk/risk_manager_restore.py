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
    snapshot = dict(data)
    snapshot.pop("last_martingale_stake", None)
    _apply_float_fields(
        manager,
        snapshot,
        ("initial_bankroll", "total_session_profit", "last_loss_stake", "dlambert_unit"),
    )
    _apply_int_fields(manager, snapshot, ("last_result_tick", "consecutive_losses_linear", "current_cooldown_ticks"))
    if "consecutive_losses_linear" not in snapshot and "consecutive_losses" in snapshot:
        manager.consecutive_losses_linear = max(0, int(snapshot["consecutive_losses"]))
    rolling = snapshot.get("rolling_wins")
    if isinstance(rolling, dict):
        manager._rolling_wins = {str(k): [int(x) for x in v] for k, v in rolling.items() if isinstance(v, list)}
    pending = snapshot.get("pending_loss")
    if isinstance(pending, dict):
        manager.pending_loss = {str(k): float(v) for k, v in pending.items()}
    if snapshot.get("last_loss_symbol") is not None:
        manager.last_loss_symbol = str(snapshot["last_loss_symbol"])
    if snapshot.get("last_loss_direction") is not None:
        manager.last_loss_direction = str(snapshot["last_loss_direction"])
