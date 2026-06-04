"""Gate de deploy: helpers e persistencia de resultado no runtime."""

import numpy as np

from src.domain.models.trade import TradeDirection


def direction_wins(direction: TradeDirection, prices: np.ndarray, index: int) -> bool:
    """Indica se a direcao prevista venceu na barra seguinte."""
    if index + 1 >= len(prices):
        return False
    up = prices[index + 1] > prices[index]
    return up if direction == TradeDirection.CALL else not up


def apply_deploy_to_runtime(
    runtime: dict,
    *,
    deploy_ok: bool,
    deploy_win_rate: float,
    val_brier: float,
) -> None:
    """Persiste resultado do gate de deploy no runtime do simbolo."""
    runtime["deploy_ok"] = bool(deploy_ok)
    runtime["deploy_win_rate"] = float(deploy_win_rate)
    if deploy_ok:
        runtime["val_brier"] = float(val_brier)
