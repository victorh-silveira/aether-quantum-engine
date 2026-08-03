"""Alinhamento de direcao meta vs price zone com trava TCN."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction_discordance import align_direction_to_rsi_trend
from src.application.services.execution_price_zone_align import align_direction_to_price_zone
from src.application.services.execution_tcn_conviction import tcn_direction_lock_active
from src.domain.models.trade import TradeDirection


def _meta_is_broken(metrics: dict[str, Any]) -> bool:
    """True quando ECE live alto e WR baixo indicam calibracao inutilizavel."""
    ece = metrics.get("live_ece")
    wr = metrics.get("live_wr")
    return bool(ece is not None and wr is not None and float(ece) > 0.5 and float(wr) < 0.3)


def align_or_keep_meta_side(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    dl_dir: TradeDirection,
    predicted_edge: float | None = None,
    meta_applied: bool = False,
) -> TradeDirection:
    """Mantem o lado meta com edge positivo; com edge <= 0 alinha a price zone."""
    if tcn_direction_lock_active(metrics):
        metrics["tcn_direction_lock"] = True
        metrics["exec_direction"] = dl_dir.name
        metrics["resolved_direction"] = dl_dir.name
        return dl_dir
    zone_side = str(metrics.get("price_zone_direction") or "").upper()
    edge_raw = metrics.get("predicted_payoff_edge", predicted_edge)
    edge_v = float(edge_raw) if edge_raw is not None else 0.0
    applied = bool(meta_applied or metrics.get("meta_classifier_applied"))
    model_broken = _meta_is_broken(metrics)
    if (
        applied
        and edge_v > 0.0
        and not model_broken
        and zone_side in {TradeDirection.CALL.name, TradeDirection.PUT.name}
        and zone_side != exec_dir.name
    ):
        metrics["price_zone_kept_meta_side"] = True
        metrics["price_zone_skipped_side"] = zone_side
        return exec_dir
    aligned = align_direction_to_price_zone(exec_dir, metrics)
    if aligned != exec_dir and applied and edge_v > 0.0 and not model_broken and aligned != dl_dir:
        metrics["price_zone_kept_meta_side"] = True
        return exec_dir
    margin_v = float(metrics.get("direction_margin", 0.0))
    if (edge_v <= 0.0 or margin_v < 0.025 or model_broken) and bool(metrics.get("rsi_trend_align_enabled", True)):
        if model_broken:
            metrics["meta_bypassed_due_to_drift"] = True
        return align_direction_to_rsi_trend(aligned, metrics)
    return aligned
