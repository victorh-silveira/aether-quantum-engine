"""Alinhamento de fusion_p_eff ao lado EXEC e Edge Cal para neg_edge."""

from __future__ import annotations

from typing import Any

from src.application.services.market_audit_log_helpers import resolve_predicted_edge


def sync_fusion_p_eff_for_direction(metrics: dict[str, Any], direction: str) -> None:
    """Alinha fusion_p_eff ao lado EXEC atual apos flip anti-loss/loss-clf."""
    if not bool(metrics.get("fusion_applied")):
        return
    side = str(direction or "").strip().upper()
    key = "fusion_p_call" if side == "CALL" else "fusion_p_put" if side == "PUT" else None
    if key is None:
        return
    raw = metrics.get(key)
    try:
        p_eff = float(raw)
    except (TypeError, ValueError):
        return
    if 0.0 < p_eff < 1.0:
        metrics["fusion_p_eff"] = p_eff


def stamp_fusion_p_eff(metrics: dict[str, Any]) -> None:
    """Grava fusion_p_eff so para telemetria; nao alimenta o gate."""
    if not bool(metrics.get("fusion_applied")):
        return
    raw = metrics.get("fusion_p_eff")
    try:
        p_eff = float(raw)
    except (TypeError, ValueError):
        return
    if 0.0 < p_eff < 1.0:
        metrics["neg_edge_fusion_p_eff"] = p_eff


def resolve_neg_side_edge(metrics: dict[str, Any], direction: str, pay: float) -> float:
    """Edge do lado pretendido (fusion_p_* do lado final, ou Cal TCN)."""
    side = str(direction or "").strip().upper()
    if bool(metrics.get("fusion_applied")) and side in {"CALL", "PUT"}:
        key = "fusion_p_call" if side == "CALL" else "fusion_p_put"
        raw = metrics.get(key)
        try:
            p = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            p = None
        if p is not None and 0.0 < p < 1.0:
            metrics["fusion_p_eff"] = p
            edge = float((p * (1.0 + pay)) - 1.0)
            metrics["neg_edge_tcn_cal_edge"] = edge
            stamp_fusion_p_eff(metrics)
            return edge
    stamp_fusion_p_eff(metrics)
    edge = float(resolve_predicted_edge(metrics, direction=side if side in {"CALL", "PUT"} else None, payout=pay))
    metrics["neg_edge_tcn_cal_edge"] = edge
    return edge
