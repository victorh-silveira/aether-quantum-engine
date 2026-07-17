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


def build_risk_state_snapshot(manager: Any) -> dict[str, Any]:
    """Monta snapshot persistivel dos campos de risco do manager."""
    return {
        "initial_bankroll": manager.initial_bankroll,
        "total_session_profit": manager.total_session_profit,
        "last_result_tick": manager.last_result_tick,
        "rolling_wins": {k: list(v) for k, v in manager._rolling_wins.items()},
        "pending_loss": dict(manager.pending_loss),
        "last_loss_stake": manager.last_loss_stake,
        "consecutive_losses_linear": manager.consecutive_losses_linear,
        "dlambert_unit": manager.dlambert_unit,
        "current_cooldown_ticks": manager.current_cooldown_ticks,
        **manager.symbol_cooldown_state(),
    }


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
