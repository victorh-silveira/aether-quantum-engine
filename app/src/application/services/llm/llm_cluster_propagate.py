"""Propagacao de decisoes LLM da ancora para indices US/EU."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_direction import cluster_direction_from_tag
from src.application.services.llm.cluster_statarb_attempt import select_cluster_symbol_attempt_order
from src.application.services.llm.cluster_statarb_select import resolve_statarb_cluster_config_for_tag
from src.application.services.llm.llm_cluster_exclusive import (
    exclusive_cluster_by_macro_enabled,
    resolve_exclusive_cluster_region,
)
from src.application.services.llm.llm_cluster_logging import log_cluster_propagation_results
from src.application.services.llm.llm_cluster_propagate_region import (
    cluster_region_active,
    propagate_cluster_region,
)
from src.application.services.llm.strategy_clusters import resolve_cluster_lists
from src.domain.models.trade import TradeDirection


def _strategy_blocks(orch: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extrai blocos strategy, correlation e macro da configuracao do orquestrador."""
    strategy = orch.config.get("strategy", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    corr = strategy.get("correlation") if isinstance(strategy.get("correlation"), dict) else {}
    macro = strategy.get("macro") if isinstance(strategy.get("macro"), dict) else {}
    return strategy, corr, macro


def _cluster_targets(strategy: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve listas de simbolos US e EU; defaults apenas se clusters ausente."""
    clusters = strategy.get("clusters") if isinstance(strategy.get("clusters"), dict) else None
    if clusters is not None:
        us_targets, eu_targets = resolve_cluster_lists(strategy)
        return tuple(us_targets), tuple(eu_targets)
    return ("OTC_NDX", "OTC_DJI"), ("OTC_FCHI", "OTC_GDAXI", "OTC_SSMI", "OTC_FTSE")


def _rolling_wr_scores(orch: Any, candidates: list[str], corr_cfg: dict[str, Any]) -> dict[str, float] | None:
    """Mapa simbolo -> WR rolling quando ha amostras suficientes."""
    if not bool(corr_cfg.get("statarb_blend_rolling_wr", True)):
        return None
    rm = getattr(orch, "risk_manager", None)
    if rm is None or not hasattr(rm, "get_wr_rolling_stats"):
        return None
    kelly = (
        orch.config.get("risk_management", {}).get("kelly", {})
        if isinstance(orch.config.get("risk_management"), dict)
        else {}
    )
    min_n = max(1, int(kelly.get("dynamic_min_samples", 10)))
    scores: dict[str, float] = {}
    for sym in candidates:
        raw = rm.get_wr_rolling_stats(sym)
        if not isinstance(raw, tuple) or len(raw) != 2:
            continue
        wr, n = raw
        if wr is not None and int(n) >= min_n:
            scores[str(sym)] = float(wr)
    return scores or None


def _cluster_allowed_sets(
    orch: Any,
    *,
    anchor_sym: str,
    metrics: dict[str, Any],
    us_targets: tuple[str, ...],
    eu_targets: tuple[str, ...],
    anchor_in_us: bool,
    anchor_in_eu: bool,
    corr_cfg: dict[str, Any],
    macro_cfg: dict[str, Any],
) -> tuple[
    TradeDirection | None,
    TradeDirection | None,
    list[str],
    list[str],
    str,
    str,
    set[str],
    set[str],
]:
    """Calcula direcoes e ordem de tentativa StatArb por cluster."""
    macro_tag = str(metrics.get("macro_sentiment") or "")
    statarb_cfg = resolve_statarb_cluster_config_for_tag(corr_cfg, macro_cfg, macro_tag)
    spreads_raw = metrics.get("statarb_spreads")
    spreads_map = spreads_raw if isinstance(spreads_raw, dict) else {}
    hmm_state = int(metrics.get("hmm_state", 0))
    us_candidates = [s for s in orch.symbols if s in us_targets and s != anchor_sym and not anchor_in_us]
    eu_candidates = [s for s in orch.symbols if s in eu_targets and s != anchor_sym and not anchor_in_eu]
    us_dir, _ = cluster_direction_from_tag(metrics.get("us_cluster"))
    eu_dir, _ = cluster_direction_from_tag(metrics.get("eu_cluster"))
    if us_dir is None:
        us_order, us_note, us_picked = [], "", set()
    else:
        us_wr = _rolling_wr_scores(orch, us_candidates, corr_cfg)
        us_order, us_note, us_picked = select_cluster_symbol_attempt_order(
            us_candidates,
            us_dir,
            spreads_map,
            hmm_state=hmm_state,
            cfg=statarb_cfg,
            wr_scores=us_wr,
        )
    if eu_dir is None:
        eu_order, eu_note, eu_picked = [], "", set()
    else:
        eu_wr = _rolling_wr_scores(orch, eu_candidates, corr_cfg)
        eu_order, eu_note, eu_picked = select_cluster_symbol_attempt_order(
            eu_candidates,
            eu_dir,
            spreads_map,
            hmm_state=hmm_state,
            cfg=statarb_cfg,
            wr_scores=eu_wr,
        )
    return us_dir, eu_dir, us_order, eu_order, us_note, eu_note, us_picked, eu_picked


def propagate_cluster_decisions(
    orch: Any,
    *,
    anchor_sym: str,
    direction: TradeDirection,
    metrics: dict[str, Any],
    decisions: dict[str, dict],
    cid: str,
) -> None:
    """Propaga decisoes para indices US/EU via tags LLM e exclusividade macro regional."""
    _ = direction
    strategy, corr_cfg, macro_cfg = _strategy_blocks(orch)
    us_targets, eu_targets = _cluster_targets(strategy)
    anchor_in_us = anchor_sym in us_targets
    anchor_in_eu = anchor_sym in eu_targets
    conviction = float(metrics.get("conviction", 0))
    exclusive = exclusive_cluster_by_macro_enabled(orch)
    active_region = resolve_exclusive_cluster_region(metrics) if exclusive else None
    macro_tag = str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")

    if exclusive and active_region is None:
        orch.logger.debug(
            "[%s] CLUSTER_SKIP || macro=%s sem cluster exclusivo (empate indefinido)",
            cid,
            macro_tag,
        )
        return

    us_dir, eu_dir, us_order, eu_order, us_note, eu_note, us_picked, eu_picked = _cluster_allowed_sets(
        orch,
        anchor_sym=anchor_sym,
        metrics=metrics,
        us_targets=us_targets,
        eu_targets=eu_targets,
        anchor_in_us=anchor_in_us,
        anchor_in_eu=anchor_in_eu,
        corr_cfg=corr_cfg,
        macro_cfg=macro_cfg,
    )
    propagated_tags: list[str] = []
    blocked_tags: list[str] = []
    inverted_tags: list[str] = []
    corrected_tags: list[str] = []
    try_alternates = bool(corr_cfg.get("statarb_try_alternate_on_block", True))
    region_kw = {
        "orch": orch,
        "metrics": metrics,
        "decisions": decisions,
        "anchor_sym": anchor_sym,
        "conviction": conviction,
        "macro_cfg": macro_cfg,
        "corr_cfg": corr_cfg,
        "active_region": active_region,
        "exclusive": exclusive,
        "macro_tag": macro_tag,
        "try_alternates": try_alternates,
    }
    if not anchor_in_us and cluster_region_active(exclusive=exclusive, active_region=active_region, region="us"):
        p, b, i, c = propagate_cluster_region(
            attempt_order=us_order,
            picked=us_picked,
            target_direction=us_dir,
            index_note=us_note,
            **region_kw,
        )
        propagated_tags.extend(p)
        blocked_tags.extend(b)
        inverted_tags.extend(i)
        corrected_tags.extend(c)
    if not anchor_in_eu and cluster_region_active(exclusive=exclusive, active_region=active_region, region="eu"):
        p, b, i, c = propagate_cluster_region(
            attempt_order=eu_order,
            picked=eu_picked,
            target_direction=eu_dir,
            index_note=eu_note,
            **region_kw,
        )
        propagated_tags.extend(p)
        blocked_tags.extend(b)
        inverted_tags.extend(i)
        corrected_tags.extend(c)

    log_cluster_propagation_results(
        orch,
        cid=cid,
        anchor_sym=anchor_sym,
        corr_cfg=corr_cfg,
        macro_tag=macro_tag,
        active_region=active_region,
        us_dir=us_dir,
        eu_dir=eu_dir,
        us_note=us_note,
        eu_note=eu_note,
        propagated_tags=propagated_tags,
        blocked_tags=blocked_tags,
        inverted_tags=inverted_tags,
        corrected_tags=corrected_tags,
    )
