"""Propagacao regional de decisoes LLM para um cluster US ou EU."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.llm_cluster_target import apply_cluster_target_decision
from src.domain.models.trade import TradeDirection


def cluster_region_active(
    *,
    exclusive: bool,
    active_region: str | None,
    region: str,
) -> bool:
    """True quando a regiao pode receber propagacao no modo exclusivo por macro."""
    if not exclusive or not active_region:
        return True
    return active_region == region


def propagate_cluster_region(
    orch: Any,
    *,
    attempt_order: list[str],
    picked: set[str],
    target_direction: TradeDirection | None,
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
    try_alternates: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Propaga direcao para indices de uma regiao com fallback quando o lider bloqueia."""
    propagated_tags: list[str] = []
    blocked_tags: list[str] = []
    inverted_tags: list[str] = []
    corrected_tags: list[str] = []
    if target_direction is None or not attempt_order:
        return propagated_tags, blocked_tags, inverted_tags, corrected_tags
    leaders = [sym for sym in attempt_order if sym in picked] or list(attempt_order)
    fallbacks = [sym for sym in attempt_order if sym not in leaders]
    single_leader_mode = len(picked) <= 1

    def _apply_symbol(target_sym: str) -> tuple[str | None, str | None, str | None, str | None]:
        """Delega aplicacao da decisao de cluster para um simbolo alvo."""
        return apply_cluster_target_decision(
            orch,
            target_sym=target_sym,
            target_direction=target_direction,
            index_note=index_note,
            metrics=metrics,
            decisions=decisions,
            anchor_sym=anchor_sym,
            conviction=conviction,
            macro_cfg=macro_cfg,
            corr_cfg=corr_cfg,
            active_region=active_region,
            exclusive=exclusive,
            macro_tag=macro_tag,
            invert_on_block=invert_on_block,
        )

    for target_sym in leaders:
        propagated, blocked, inverted, corrected = _apply_symbol(target_sym)
        if propagated:
            propagated_tags.append(propagated)
            if corrected:
                corrected_tags.append(corrected)
            if single_leader_mode:
                return propagated_tags, blocked_tags, inverted_tags, corrected_tags
        elif inverted:
            inverted_tags.append(inverted)
            if single_leader_mode:
                return propagated_tags, blocked_tags, inverted_tags, corrected_tags
        elif blocked:
            blocked_tags.append(blocked)
    if try_alternates and single_leader_mode and not propagated_tags and not inverted_tags:
        for target_sym in fallbacks:
            propagated, blocked, inverted, corrected = _apply_symbol(target_sym)
            if propagated:
                propagated_tags.append(propagated)
                if corrected:
                    corrected_tags.append(corrected)
                return propagated_tags, blocked_tags, inverted_tags, corrected_tags
            if inverted:
                inverted_tags.append(inverted)
                return propagated_tags, blocked_tags, inverted_tags, corrected_tags
            if blocked:
                blocked_tags.append(blocked)
    return propagated_tags, blocked_tags, inverted_tags, corrected_tags
