"""Condicoes para permitir martingale em recovery com metricas Deep Learning."""


def _martingale_raw_side(conviction: float, dl_metrics: dict | None) -> float:
    """Retorna conviccao bruta maxima (CALL ou PUT) a partir de metricas DL ou score."""
    if isinstance(dl_metrics, dict) and dl_metrics.get("raw_prob") is not None:
        raw = float(dl_metrics["raw_prob"])
        return max(raw, 1.0 - raw)
    return float(conviction)


def martingale_repeat_loss_blocked(
    symbol: str,
    order_direction: str | None,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """True se martingale deve ser bloqueado por repetir symbol+direction da ultima loss."""
    if not order_direction or not last_loss_direction:
        return False
    return last_loss_symbol == symbol and str(order_direction) == str(last_loss_direction)


def _martingale_raw_block(
    conviction: float,
    dl_metrics: dict | None,
    recovery_martingale_min_raw: float,
) -> str | None:
    """Bloqueia martingale quando raw side fica abaixo do minimo configurado."""
    min_raw = float(recovery_martingale_min_raw)
    if min_raw <= 0.0:
        return None
    raw_side = _martingale_raw_side(conviction, dl_metrics)
    if raw_side + 1e-9 < min_raw:
        return "raw_conviction"
    return None


def _martingale_conviction_block(
    conviction: float,
    recovery_threshold: float,
    recovery_martingale_min_conviction: float,
    *,
    force_on_pending_loss: bool,
) -> str | None:
    """Bloqueia martingale por conviccao quando force_on_pending_loss esta desligado."""
    if force_on_pending_loss:
        return None
    min_conv = float(recovery_martingale_min_conviction)
    if conviction + 1e-9 < min_conv:
        return "conviction"
    if conviction + 1e-9 < float(recovery_threshold):
        return "recovery_threshold"
    return None


def martingale_block_reason(
    *,
    pending_loss: dict[str, float],
    recovery_threshold: float,
    conviction: float,
    symbol: str,
    dl_metrics: dict | None = None,
    max_val_brier: float = 0.28,
    order_direction: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    recovery_martingale_min_conviction: float = 0.45,
    recovery_martingale_min_raw: float = 0.0,
    force_on_pending_loss: bool = True,
) -> str | None:
    """Retorna motivo de bloqueio do martingale ou None se permitido."""
    loss_to_recover = sum(pending_loss.values())
    if loss_to_recover <= 0.0:
        return "no_pending_loss"
    raw_block = _martingale_raw_block(conviction, dl_metrics, recovery_martingale_min_raw)
    if raw_block:
        return raw_block
    conv_block = _martingale_conviction_block(
        conviction,
        recovery_threshold,
        recovery_martingale_min_conviction,
        force_on_pending_loss=force_on_pending_loss,
    )
    if conv_block:
        return conv_block
    dl_recovery = loss_to_recover > 0.0 and force_on_pending_loss
    if isinstance(dl_metrics, dict) and martingale_dl_metrics_block(
        dl_metrics,
        max_val_brier=max_val_brier,
        recovery_pending=dl_recovery,
    ):
        return "dl_metrics"
    if martingale_repeat_loss_blocked(symbol, order_direction, last_loss_symbol, last_loss_direction):
        return "repeat_loss"
    return None


def martingale_allowed(
    *,
    pending_loss: dict[str, float],
    recovery_threshold: float,
    conviction: float,
    symbol: str,
    dl_metrics: dict | None = None,
    max_val_brier: float = 0.28,
    order_direction: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    recovery_martingale_min_conviction: float = 0.45,
    recovery_martingale_min_raw: float = 0.0,
    force_on_pending_loss: bool = True,
) -> bool:
    """Indica se martingale pode ser usado neste ciclo de recovery."""
    return (
        martingale_block_reason(
            pending_loss=pending_loss,
            recovery_threshold=recovery_threshold,
            conviction=conviction,
            symbol=symbol,
            dl_metrics=dl_metrics,
            max_val_brier=max_val_brier,
            order_direction=order_direction,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
            recovery_martingale_min_conviction=recovery_martingale_min_conviction,
            recovery_martingale_min_raw=recovery_martingale_min_raw,
            force_on_pending_loss=force_on_pending_loss,
        )
        is None
    )


def apply_win_to_pending_loss(pending_loss: dict[str, float], profit: float) -> None:
    """Reduz perdas pendentes com lucro parcial de um contrato."""
    remaining_profit = profit
    for sym in list(pending_loss.keys()):
        if remaining_profit <= 0:
            break
        current_loss = pending_loss[sym]
        if current_loss <= remaining_profit:
            remaining_profit -= current_loss
            pending_loss[sym] = 0.0
        else:
            pending_loss[sym] = current_loss - remaining_profit
            remaining_profit = 0.0


def martingale_dl_metrics_block(metrics: dict, *, max_val_brier: float, recovery_pending: bool = False) -> bool:
    """True se metricas DL impedem martingale (gate, brier ou deploy)."""
    if recovery_pending:
        return False
    if metrics.get("gate_reason"):
        return True
    val_brier = float(metrics.get("val_brier", 1.0))
    if val_brier + 1e-9 >= max_val_brier:
        return True
    return metrics.get("deploy_ok") is False
