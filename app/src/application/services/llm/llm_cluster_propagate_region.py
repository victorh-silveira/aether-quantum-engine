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


def _merge_apply_result(
    propagated_tags: list[str],
    blocked_tags: list[str],
    inverted_tags: list[str],
    corrected_tags: list[str],
    propagated: str | None,
    blocked: str | None,
    inverted: str | None,
    corrected: str | None,
) -> bool:
    """Acumula tags de propagacao e indica se houve hit propagado ou invertido."""
    if propagated:
        propagated_tags.append(propagated)
        if inverted:
            inverted_tags.append(inverted)
        if corrected:
            corrected_tags.append(corrected)
        return True
    if inverted:
        inverted_tags.append(inverted)
        return True
    if blocked:
        blocked_tags.append(blocked)
    return False


def _try_symbol_batch(
    symbols: list[str],
    apply_symbol: Any,
    *,
    stop_on_first_hit: bool,
) -> tuple[list[str], list[str], list[str], list[str], bool]:
    """Aplica decisao em lote e retorna tags acumuladas e flag de hit."""
    propagated_tags: list[str] = []
    blocked_tags: list[str] = []
    inverted_tags: list[str] = []
    corrected_tags: list[str] = []
    hit = False
    for target_sym in symbols:
        propagated, blocked, inverted, corrected = apply_symbol(target_sym)
        if _merge_apply_result(
            propagated_tags,
            blocked_tags,
            inverted_tags,
            corrected_tags,
            propagated,
            blocked,
            inverted,
            corrected,
        ):
            hit = True
            if stop_on_first_hit:
                return propagated_tags, blocked_tags, inverted_tags, corrected_tags, hit
    return propagated_tags, blocked_tags, inverted_tags, corrected_tags, hit


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
    try_alternates: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Propaga direcao para indices de uma regiao com fallback quando o lider bloqueia."""
    empty: tuple[list[str], list[str], list[str], list[str]] = ([], [], [], [])
    if target_direction is None or not attempt_order:
        return empty
    if bool(corr_cfg.get("statarb_require_z_align", False)) and not picked:
        return empty
    leaders = [sym for sym in attempt_order if sym in picked] or list(attempt_order)
    fallbacks = [sym for sym in attempt_order if sym not in leaders]
    single_leader_mode = len(picked) <= 1

    def apply_symbol(target_sym: str) -> tuple[str | None, str | None, str | None, str | None]:
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
        )

    propagated_tags, blocked_tags, inverted_tags, corrected_tags, hit = _try_symbol_batch(
        leaders,
        apply_symbol,
        stop_on_first_hit=single_leader_mode,
    )
    if hit and single_leader_mode:
        return propagated_tags, blocked_tags, inverted_tags, corrected_tags
    if try_alternates and single_leader_mode and not propagated_tags and not inverted_tags and fallbacks:
        alt_p, alt_b, alt_i, alt_c, _ = _try_symbol_batch(
            fallbacks,
            apply_symbol,
            stop_on_first_hit=True,
        )
        return alt_p, alt_b, alt_i, alt_c
    return propagated_tags, blocked_tags, inverted_tags, corrected_tags
