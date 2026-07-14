"""Gate de deploy: helpers e persistencia de resultado no runtime."""

from src.application.services.deep_learning.dl_labels import LabelSpec, binary_label_at_index
from src.domain.models.trade import TradeDirection


def direction_wins(
    direction: TradeDirection,
    prices,
    index: int,
    *,
    label_horizon_bars: int = 1,
    label_spec: LabelSpec | None = None,
) -> bool:
    """Indica se a direcao prevista venceu no horizonte de label configurado."""
    spec = label_spec or LabelSpec(horizon_bars=label_horizon_bars)
    target_up = binary_label_at_index(
        prices,
        index,
        spec.horizon_bars,
        smooth_bars=spec.smooth_bars,
        label_mode=spec.label_mode,
        ma_window=spec.ma_window,
    )
    return target_up if direction == TradeDirection.CALL else not target_up


def call_target_label(
    prices,
    index: int,
    *,
    label_spec: LabelSpec,
) -> float:
    """Retorna 1.0 quando o label indica CALL no horizonte configurado."""
    target_up = binary_label_at_index(
        prices,
        index,
        label_spec.horizon_bars,
        smooth_bars=label_spec.smooth_bars,
        label_mode=label_spec.label_mode,
        ma_window=label_spec.ma_window,
    )
    return 1.0 if target_up else 0.0


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
