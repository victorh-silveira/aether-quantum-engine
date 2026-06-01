"""Aplicacao de decisao LLM em um simbolo do cluster."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_statarb_direction import correct_cluster_direction_for_tag
from src.application.services.llm.duration_logic import enforce_minimum_duration
from src.application.services.llm.llm_bridge_telemetry import store_symbol_decision
from src.application.services.llm.llm_cluster_guards import cluster_execute_block_reason
from src.application.services.llm.llm_cluster_invert import (
    apply_cluster_binary_invert,
    cluster_invert_llm_side_enabled,
    flip_binary_direction,
)
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
) -> tuple[str | None, str | None, str | None, str | None]:
    """Aplica decisao em um simbolo do cluster e retorna tags de log."""
    region_note = f" region={active_region} macro={macro_tag}" if active_region else ""
    target_metrics = metrics.copy()
    target_metrics["cluster_target_sym"] = target_sym
    original_direction = target_direction
    inverted_tag = None
    if cluster_invert_llm_side_enabled(corr_cfg):
        alt = flip_binary_direction(target_direction)
        if alt is not None:
            target_direction = alt
            target_metrics["llm_exec_inverted"] = True
            target_metrics["decision_source"] = "cluster_invert"
            inverted_tag = f"{target_sym}[{original_direction.name[:1]}->{target_direction.name[:1]}]"
    corrected, did_correct, correct_note = correct_cluster_direction_for_tag(
        target_direction,
        macro_tag=macro_tag,
        target_sym=target_sym,
        metrics=target_metrics,
        corr_cfg=corr_cfg,
        macro_cfg=macro_cfg,
    )
    if did_correct:
        target_direction = corrected
        target_metrics["llm_statarb_dir_corrected"] = True
        target_metrics["decision_source"] = "cluster_statarb_dir"
        if active_region == "us":
            target_metrics["us_cluster"] = target_direction.name
        elif active_region == "eu":
            target_metrics["eu_cluster"] = target_direction.name
    execute_reason = cluster_execute_block_reason(
        orch,
        target_metrics,
        conviction,
        target_direction,
        macro_cfg,
        corr_cfg,
        active_region=active_region,
        target_sym=target_sym,
        llm_cluster_explicit=True,
        index_note=index_note,
    )
    target_metrics["execute"] = execute_reason == "allowed"
    target_metrics["llm_block_reason"] = execute_reason
    corrected_tag = None
    if did_correct:
        corrected_tag = f"{target_sym}[{original_direction.name[:1]}->{target_direction.name[:1]}]"
    # Remoção total da inversão lateral em clusters para garantir a direcionalidade gerada pela LLM
    can_invert = False

    if not target_metrics.get("llm_exec_inverted"):
        note_prefix = f"{correct_note} | " if did_correct else ""
        target_metrics["llm_note"] = (
            f"{note_prefix}CLUSTER_TAG ({target_direction.name}) conv={conviction:.1%}{region_note} | {index_note} from {anchor_sym}"
        )
        if not did_correct and not target_metrics.get("llm_exec_inverted"):
            target_metrics["decision_source"] = "cluster_regime"
    target_metrics["cluster_active_region"] = active_region or ""
    target_metrics["cluster_exclusive_macro"] = exclusive
    target_metrics["duration"] = enforce_minimum_duration(target_sym, target_metrics.get("duration", 15))
    store_symbol_decision(decisions, target_sym, target_direction, target_metrics)
    reason = str(target_metrics.get("llm_block_reason") or "blocked")
    sym_tag = f"{target_sym}[{target_direction.name[:1]}]"
    if target_metrics["execute"]:
        return sym_tag, None, inverted_tag, corrected_tag
    return None, f"{sym_tag}:{reason}", inverted_tag, corrected_tag
