"""Pisos de conviccao para entradas em recovery financeiro."""

from src.domain.risk.kelly_runtime_config import kelly_runtime_from_config, load_kelly_runtime_from_settings
from src.domain.risk.stake_sizing import metric_float, raw_side_from_metrics


def _merged_config(kelly_config: dict, dlambert_config: dict) -> dict:
    """Resolve ou aplica  merged config."""
    merged = dict(kelly_config)
    merged.update(dlambert_config)
    return merged


def _runtime(kelly_config: dict) -> dict:
    """Resolve ou aplica  runtime."""
    if isinstance(kelly_config, dict) and "recovery_conviction_ladder" in kelly_config:
        try:
            return kelly_runtime_from_config({"kelly": kelly_config})
        except ValueError:
            pass
    return load_kelly_runtime_from_settings()


def scaled_recovery_min_val_accuracy(kelly_config: dict, *, consecutive_losses: int = 0) -> float:
    """Piso de val_accuracy para recovery; sobe com perdas lineares."""
    base_val = float(kelly_config.get("recovery_min_val_accuracy", 0.53))
    losses = int(consecutive_losses)
    if losses >= 4:
        return max(base_val, 0.55)
    if losses >= 3:
        return max(base_val, 0.54)
    if losses == 2:
        return max(base_val, 0.53)
    return base_val


def recovery_min_conviction(
    kelly_config: dict,
    dlambert_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses_linear: int = 0,
) -> float:
    """Resolve ou aplica recovery min conviction."""
    cfg = _merged_config(kelly_config, dlambert_config)
    runtime = _runtime(kelly_config)
    if "recovery_sizing_conviction" in cfg and float(cfg["recovery_sizing_conviction"]) > 0.0:
        min_conv = float(cfg["recovery_sizing_conviction"])
    elif "recovery_min_conviction" in cfg and float(cfg["recovery_min_conviction"]) > 0.0:
        min_conv = float(cfg["recovery_min_conviction"])
    else:
        min_conv = float(runtime["recovery_sizing_conviction"])
    pending = sum(float(v) for v in pending_loss.values())
    force_min = float(cfg["recovery_min_conviction"]) if "recovery_min_conviction" in cfg else min_conv
    if "recovery_force_pending_min" in kelly_config:
        force_pending = float(kelly_config["recovery_force_pending_min"])
    else:
        force_pending = float(runtime["recovery_force_pending_min"])
    if force_pending > 0.0 and pending + 1e-9 >= force_pending:
        min_conv = min(min_conv, force_min)
    ladder = runtime["recovery_conviction_ladder"]
    losses = int(consecutive_losses_linear)
    if losses >= 2:
        min_conv = max(min_conv, float(ladder["losses_2"]))
    elif losses == 1:
        min_conv = max(min_conv, float(ladder["losses_1"]))
    if losses >= 3:
        min_conv = max(min_conv, float(ladder["losses_3"]))
    if losses >= 4:
        min_conv = max(min_conv, float(ladder["losses_4"]))
    return min_conv


def recovery_dl_conviction_ok(
    dl_metrics: dict,
    kelly_config: dict,
    dlambert_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses_linear: int = 0,
) -> bool:
    """Resolve ou aplica recovery dl conviction ok."""
    if dl_metrics.get("deploy_ok") is False:
        return False
    cfg = _merged_config(kelly_config, dlambert_config)
    runtime = _runtime(kelly_config)
    min_conv = recovery_min_conviction(
        kelly_config,
        dlambert_config,
        pending_loss=pending_loss,
        consecutive_losses_linear=consecutive_losses_linear,
    )
    min_val = scaled_recovery_min_val_accuracy(
        cfg if "recovery_min_val_accuracy" in cfg else runtime,
        consecutive_losses=int(consecutive_losses_linear),
    )
    score = metric_float(dl_metrics, "trade_score", "conviction", "calibrated_prob", default=0.0)
    raw_side = raw_side_from_metrics(dl_metrics)
    val = metric_float(dl_metrics, "val_accuracy", default=0.0)
    if min_val > 0.0 and val + 1e-9 < min_val:
        return False
    effective = max(score, raw_side) if score >= 0.50 else score
    if effective + 1e-9 >= min_conv:
        return True
    return score < 1e-9 and raw_side + 1e-9 >= min_conv


def recovery_dl_entry_allowed(
    dl_metrics: dict,
    kelly_config: dict,
    dlambert_config: dict,
    *,
    pending_loss: dict[str, float],
    consecutive_losses_linear: int = 0,
    recovery_forced: bool = False,
) -> bool:
    """Resolve ou aplica recovery dl entry allowed."""
    if dl_metrics.get("deploy_ok") is False:
        return False
    if recovery_forced or dl_metrics.get("recovery_forced"):
        return True
    cfg = _merged_config(kelly_config, dlambert_config)
    runtime = _runtime(kelly_config)
    min_val = scaled_recovery_min_val_accuracy(
        cfg if "recovery_min_val_accuracy" in cfg else runtime,
        consecutive_losses=int(consecutive_losses_linear),
    )
    val = metric_float(dl_metrics, "val_accuracy", default=0.0)
    if min_val > 0.0 and val + 1e-9 < min_val:
        return False
    return recovery_dl_conviction_ok(
        dl_metrics,
        kelly_config,
        dlambert_config,
        pending_loss=pending_loss,
        consecutive_losses_linear=consecutive_losses_linear,
    )
