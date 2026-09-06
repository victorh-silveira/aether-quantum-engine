"""Follow M5: alinh EXEC a vela fechada quando Edge Cal do lado da vela >= piso."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_fusion_p_eff import sync_fusion_p_eff_for_direction
from src.application.services.execution_signal_skip import apply_kelly_soft
from src.application.services.market_audit_log_helpers import resolve_predicted_edge
from src.domain.models.trade import TradeDirection


def _min_edge_follow(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Piso Edge explore/recovery para follow de vela."""
    for src in (metrics, cfg):
        if not isinstance(src, dict):
            continue
        raw = src.get("min_edge_explore")
        if raw is None:
            raw = src.get("min_edge_recovery")
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            return val
    return 0.015


def candle_follow_edge_ok(
    metrics: dict[str, Any],
    candle: str,
    *,
    cfg: dict[str, Any],
) -> bool:
    """True se Edge Cal do lado da vela >= piso Soft/neg_edge."""
    min_edge = _min_edge_follow(metrics, cfg)
    edge = float(resolve_predicted_edge(metrics, direction=candle))
    metrics["micro_discord_follow_candle_edge"] = edge
    metrics["micro_discord_follow_min_edge"] = min_edge
    if edge + 1e-12 < min_edge:
        metrics["micro_discord_follow_blocked"] = "edge_nonpos" if edge + 1e-12 <= 0.0 else "edge_subfloor"
        return False
    metrics.pop("micro_discord_follow_blocked", None)
    return True


def apply_micro_discord_follow_candle(
    metrics: dict[str, Any],
    *,
    candle: str,
    exec_side: str,
    body: float,
    cfg: dict[str, Any],
) -> bool:
    """Alinha EXEC a vela + Soft_SIZE; True se aplicou follow."""
    if not bool(cfg.get("micro_discord_follow_candle", False)):
        return False
    if not candle_follow_edge_ok(metrics, candle, cfg=cfg):
        return False
    new_side = TradeDirection[candle]
    metrics["exec_direction"] = new_side.name
    metrics["resolved_direction"] = new_side.name
    sync_fusion_p_eff_for_direction(metrics, new_side.name)
    soft_mult = float(cfg.get("micro_discord_follow_kelly_mult", cfg.get("anti_loss_soft_kelly_mult", 0.55)))
    apply_kelly_soft(
        metrics,
        soft_mult,
        waived="micro_discord_follow",
        flag="micro_discord_follow_soft",
    )
    metrics["micro_discord_followed"] = True
    metrics["micro_discord_follow_from"] = exec_side
    metrics["micro_discord_candle"] = candle
    metrics["micro_discord_exec"] = new_side.name
    metrics["micro_discord_body"] = float(body)
    metrics["micro_discord_confirmed"] = True
    return True
