"""Inversao binaria PUT/CALL quando o cluster nao executa no lado LLM."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def flip_binary_direction(direction: TradeDirection) -> TradeDirection | None:
    """Retorna CALL/PUT oposto ou None para direcoes nao binarias."""
    if direction == TradeDirection.CALL:
        return TradeDirection.PUT
    if direction == TradeDirection.PUT:
        return TradeDirection.CALL
    return None


def cluster_invert_on_block_enabled(corr_cfg: dict[str, Any]) -> bool:
    """Indica se o cluster deve inverter direcao quando bloqueado."""
    return bool(corr_cfg.get("cluster_invert_on_block", True))


def cluster_invert_llm_side_enabled(corr_cfg: dict[str, Any]) -> bool:
    """Indica se executa o lado oposto ao CALL/PUT propagado da LLM."""
    return bool(corr_cfg.get("cluster_invert_llm_side", False))




def apply_cluster_binary_invert(
    target_direction: TradeDirection,
    target_metrics: dict[str, Any],
    *,
    index_note: str,
    anchor_sym: str,
    region_note: str,
    conviction: float,
) -> tuple[TradeDirection, dict[str, Any], bool]:
    """Inverte PUT/CALL e forca execute quando o lado original foi vetado."""
    alt = flip_binary_direction(target_direction)
    if alt is None:
        return target_direction, target_metrics, False
    inv_conv = max(0.0, min(0.99, max(float(conviction), 1.0 - float(conviction))))
    out = dict(target_metrics)
    out["execute"] = True
    out["llm_exec_inverted"] = True
    out["conviction"] = inv_conv
    out["llm_note"] = (
        f"CLUSTER_INVERT {target_direction.name}->{alt.name} conv={inv_conv:.1%}{region_note}"
        f" | {index_note} from {anchor_sym}"
    )
    out["decision_source"] = "cluster_invert"
    return alt, out, True
