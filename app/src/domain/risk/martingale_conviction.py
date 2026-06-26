"""Pisos de conviccao para martingale em recovery."""

from src.domain.risk.stake_sizing import raw_side_from_metrics


def martingale_min_conviction(
    kelly_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses: int = 0,
) -> float:
    """Resolve piso de conviccao para martingale, escalonando com perdas consecutivas."""
    min_conv = float(kelly_config.get("martingale_sizing_conviction", 0.58))
    if min_conv <= 0.0:
        min_conv = float(kelly_config.get("recovery_martingale_min_conviction", 0.58))
    if min_conv <= 0.0:
        min_conv = 0.58
    pending = sum(float(v) for v in pending_loss.values())
    force_min = float(kelly_config.get("recovery_martingale_min_conviction", min_conv))
    force_pending = float(kelly_config.get("recovery_force_pending_min", 0.0))
    if force_pending > 0.0 and pending + 1e-9 >= force_pending:
        min_conv = min(min_conv, force_min)
    losses = int(consecutive_losses)
    if losses >= 2:
        min_conv = max(min_conv, 0.60)
    elif losses == 1:
        min_conv = max(min_conv, 0.58)
    if losses >= 3:
        min_conv = max(min_conv, 0.62)
    if losses >= 4:
        min_conv = max(min_conv, 0.64)
    return min_conv


def martingale_dl_conviction_ok(
    dl_metrics: dict,
    kelly_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses: int = 0,
) -> bool:
    """Exige piso de sinal e val_accuracy para martingale com metricas DL."""
    if dl_metrics.get("deploy_ok") is False:
        return False
    min_conv = martingale_min_conviction(
        kelly_config,
        pending_loss=pending_loss,
        consecutive_losses=consecutive_losses,
    )
    min_val = float(kelly_config.get("martingale_min_val_accuracy", 0.50))
    score = float(dl_metrics.get("trade_score", dl_metrics.get("conviction", 0.0)))
    raw_side = raw_side_from_metrics(dl_metrics)
    val = float(dl_metrics.get("val_accuracy", 0.0))
    if min_val > 0.0 and val + 1e-9 < min_val:
        return False
    effective = max(score, raw_side)
    if effective + 1e-9 >= min_conv:
        return True
    return score < 1e-9 and raw_side + 1e-9 >= min_conv


def martingale_dl_entry_allowed(
    dl_metrics: dict,
    kelly_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses: int = 0,
    recovery_forced: bool = False,
) -> bool:
    """Valida metricas DL antes de liberar stake de martingale."""
    if dl_metrics.get("deploy_ok") is False:
        return False
    if recovery_forced or dl_metrics.get("recovery_forced"):
        return True
    min_val = float(kelly_config.get("martingale_min_val_accuracy", 0.50))
    val = float(dl_metrics.get("val_accuracy", 0.0))
    if min_val > 0.0 and val + 1e-9 < min_val:
        return False
    return martingale_dl_conviction_ok(
        dl_metrics,
        kelly_config,
        pending_loss=pending_loss,
        consecutive_losses=consecutive_losses,
    )
