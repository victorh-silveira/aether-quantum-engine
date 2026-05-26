"""Aplicacao de decisao LLM em um simbolo do cluster."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.duration_logic import enforce_minimum_duration
from src.application.services.llm.llm_bridge_telemetry import store_symbol_decision
from src.application.services.llm.llm_cluster_guards import cluster_execute_flag
from src.application.services.llm.llm_cluster_invert import apply_cluster_binary_invert
from src.domain.models.trade import TradeDirection


def apply_cluster_target_decision(
    orch: Any,
    *,
    target_sym: str,
    target_direction: TradeDirection,
    index_note: str,
    metrics: dict[str, Any],
    decisions: dict[str, dict],
    anchor_sym: str,
    conviction: float,
    macro_cfg: dict[str, Any],
    corr_cfg: dict[str, Any],
    active_region: str | None,
    exclusive: bool,
    macro_tag: str,
    invert_on_block: bool,
) -> tuple[str | None, str | None, str | None]:
    """Aplica decisao em um simbolo do cluster e retorna tags de log."""
    region_note = f" region={active_region} macro={macro_tag}" if active_region else ""
    target_metrics = metrics.copy()
    original_direction = target_direction
    target_metrics["execute"] = cluster_execute_flag(
        orch,
        target_metrics,
        conviction,
        target_direction,
        macro_cfg,
        corr_cfg,
        active_region=active_region,
        target_sym=target_sym,
        llm_cluster_explicit=True,
    )
    inverted_tag = None
    if not target_metrics["execute"] and invert_on_block:
        target_direction, target_metrics, did_invert = apply_cluster_binary_invert(
            target_direction,
            target_metrics,
            index_note=index_note,
            anchor_sym=anchor_sym,
            region_note=region_note,
            conviction=conviction,
        )
        if did_invert:
            inverted_tag = f"{target_sym}[{original_direction.name[:1]}->{target_direction.name[:1]}]"
    if not target_metrics.get("llm_exec_inverted"):
        target_metrics["llm_note"] = (
            f"CLUSTER_TAG ({target_direction.name}) conv={conviction:.1%}{region_note} | {index_note} from {anchor_sym}"
        )
        target_metrics["decision_source"] = "cluster_regime"
    target_metrics["cluster_active_region"] = active_region or ""
    target_metrics["cluster_exclusive_macro"] = exclusive
    target_metrics["duration"] = enforce_minimum_duration(target_sym, target_metrics.get("duration", 15))
    store_symbol_decision(decisions, target_sym, target_direction, target_metrics)
    sym_tag = f"{target_sym}[{target_direction.name[:1]}]"
    if target_metrics["execute"]:
        return sym_tag, None, inverted_tag
    return None, sym_tag, inverted_tag
