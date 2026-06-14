"""Gate de deploy: helpers e persistencia de resultado no runtime."""

from src.application.services.deep_learning.dl_labels import binary_label_at_index
from src.domain.models.trade import TradeDirection


def direction_wins(
    direction: TradeDirection,
    prices,
    index: int,
    *,
    label_horizon_bars: int = 1,
) -> bool:
    """Indica se a direcao prevista venceu no horizonte de label configurado."""
    target_up = binary_label_at_index(prices, index, label_horizon_bars)
    return target_up if direction == TradeDirection.CALL else not target_up


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
