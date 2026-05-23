"""Propagacao de decisoes LLM da ancora para indices US/EU."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_direction import cluster_direction_from_tag
from src.application.services.llm.cluster_statarb_select import (
    resolve_statarb_cluster_config,
    select_cluster_symbols_by_statarb,
)
from src.application.services.llm.duration_logic import enforce_minimum_duration
from src.application.services.llm.llm_bridge_telemetry import store_symbol_decision
from src.application.services.llm.llm_cluster_exclusive import (
    cluster_region_for_symbol,
    exclusive_cluster_by_macro_enabled,
    resolve_exclusive_cluster_region,
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
) -> tuple[TradeDirection | None, TradeDirection | None, set[str], set[str], str, str]:
    """Calcula direcoes e indices permitidos por cluster via tags LLM e StatArb."""
    statarb_cfg = resolve_statarb_cluster_config(corr_cfg, macro_cfg)
    spreads_raw = metrics.get("statarb_spreads")
    spreads_map = spreads_raw if isinstance(spreads_raw, dict) else {}
    hmm_state = int(metrics.get("hmm_state", 0))
    us_candidates = [s for s in orch.symbols if s in us_targets and s != anchor_sym and not anchor_in_us]
    eu_candidates = [s for s in orch.symbols if s in eu_targets and s != anchor_sym and not anchor_in_eu]
    us_dir, _ = cluster_direction_from_tag(metrics.get("us_cluster"))
    eu_dir, _ = cluster_direction_from_tag(metrics.get("eu_cluster"))
    if us_dir is None:
        us_allowed, us_note = set(), ""
    else:
        us_allowed, us_note = select_cluster_symbols_by_statarb(
            us_candidates, us_dir, spreads_map, hmm_state=hmm_state, cfg=statarb_cfg
        )
    if eu_dir is None:
        eu_allowed, eu_note = set(), ""
    else:
        eu_allowed, eu_note = select_cluster_symbols_by_statarb(
            eu_candidates, eu_dir, spreads_map, hmm_state=hmm_state, cfg=statarb_cfg
        )
    return us_dir, eu_dir, us_allowed, eu_allowed, us_note, eu_note


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

    us_dir, eu_dir, us_allowed, eu_allowed, us_note, eu_note = _cluster_allowed_sets(
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

    for target_sym in orch.symbols:
        if target_sym == anchor_sym:
            continue
        sym_region = cluster_region_for_symbol(target_sym, us_targets=us_targets, eu_targets=eu_targets)
        if exclusive and active_region and sym_region and sym_region != active_region:
            continue
        if target_sym in us_targets and not anchor_in_us:
            target_direction, index_note = us_dir, us_note
            allowed = us_allowed
        elif target_sym in eu_targets and not anchor_in_eu:
            target_direction, index_note = eu_dir, eu_note
            allowed = eu_allowed
        else:
            continue
        if target_direction is None or target_sym not in allowed:
            continue

        region_note = f" region={active_region} macro={macro_tag}" if active_region else ""
        target_metrics = metrics.copy()
        target_metrics["llm_note"] = (
            f"CLUSTER_TAG ({target_direction.name}) conv={conviction:.1%}{region_note} | {index_note} from {anchor_sym}"
        )
        target_metrics["decision_source"] = "cluster_regime"
        target_metrics["cluster_active_region"] = active_region or ""
        target_metrics["cluster_exclusive_macro"] = exclusive
        target_metrics["duration"] = enforce_minimum_duration(target_sym, target_metrics.get("duration", 15))
        store_symbol_decision(decisions, target_sym, target_direction, target_metrics)
        propagated_tags.append(f"{target_sym}[{target_direction.name[:1]}]")

    if propagated_tags:
        orch.logger.debug(
            "[%s] CORR CLUSTER || %s [%s] >> [%s]",
            cid,
            anchor_sym,
            direction.name,
            ", ".join(propagated_tags),
        )
