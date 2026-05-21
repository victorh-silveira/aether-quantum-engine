"""Montagem de MacroSnapshot e fallback M5 para clusters flat em M15."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.global_macro_confluence import (
    aggregate_cluster_vote,
    classify_transatlantic_confluence,
    eurusd_bias_from_confluence,
    expected_cluster_tags_line,
    format_macro_confluence_block,
    fx_reference_context_line,
)
from src.application.services.llm.macro_config import ClusterVote, MacroSnapshot, resolve_macro_config


def macro_snapshot_from_votes(
    us_vote: ClusterVote,
    eu_vote: ClusterVote,
    macro_cfg: dict[str, Any] | None = None,
    *,
    cluster_suffix: str = "",
    statarb_spreads: dict[str, float] | None = None,
    hmm_state: int = 0,
    hmm_prob: float = 1.0,
) -> MacroSnapshot:
    """Monta MacroSnapshot a partir de votos US/EU ja agregados."""
    cfg = resolve_macro_config(macro_cfg)
    tag = classify_transatlantic_confluence(us_vote.direction, eu_vote.direction)
    bias = eurusd_bias_from_confluence(tag, us_dir=us_vote.direction, eu_dir=eu_vote.direction)
    us_summary = f"US_CLUSTER [{', '.join(us_vote.parts)}]"
    eu_summary = f"EU_CLUSTER [{', '.join(eu_vote.parts)}]"
    fx_ref = fx_reference_context_line(tag, cfg.get("fx_reference_pairs"))
    cluster_status = f"{us_summary} || {eu_summary}"
    cluster_quant_line = expected_cluster_tags_line(
        tag=tag,
        us_dir=us_vote.direction,
        eu_dir=eu_vote.direction,
        us_strength=us_vote.strength,
        eu_strength=eu_vote.strength,
        macro_cfg=cfg,
    )

    # Append HMM regime information to telemetry if active
    if hmm_prob < 1.0 or hmm_state > 0:
        regime_lbl = "MEAN_REVERSION" if hmm_state == 0 else "TRENDING"
        cluster_quant_line = f"{cluster_quant_line} | HMM_regime={regime_lbl} ({hmm_prob * 100:.1f}%)"

    suffix = (cluster_suffix or "").strip()
    if suffix:
        cluster_quant_line = f"{cluster_quant_line} | {suffix}".strip(" |")
    macro_block = format_macro_confluence_block(
        us_summary,
        eu_summary,
        tag,
        fx_ref,
        eurusd_bias=bias,
        cluster_quant_line=cluster_quant_line,
    )
    return MacroSnapshot(
        us_dir=us_vote.direction,
        eu_dir=eu_vote.direction,
        us_strength=us_vote.strength,
        eu_strength=eu_vote.strength,
        tag=tag,
        eurusd_bias=bias,
        cluster_status=cluster_status,
        macro_block=macro_block,
        fx_reference_line=fx_ref,
        us_parts=us_vote.parts,
        eu_parts=eu_vote.parts,
        statarb_spreads=statarb_spreads,
        hmm_state=hmm_state,
        hmm_prob=hmm_prob,
    )


def build_macro_snapshot(
    us_symbols: list[str],
    eu_symbols: list[str],
    closes_map: dict[str, list[float]],
    macro_cfg: dict[str, Any] | None = None,
    *,
    statarb_spreads: dict[str, float] | None = None,
    hmm_state: int = 0,
    hmm_prob: float = 1.0,
) -> MacroSnapshot:
    """Monta snapshot macro completo a partir de fechamentos dos clusters."""
    cfg = resolve_macro_config(macro_cfg)
    threshold = float(cfg["cluster_return_threshold_pct"])
    min_move = float(cfg["cluster_min_move_pct"])
    min_idx = int(cfg["min_indices_for_vote"])
    label_bundle = {"cluster_labels": cfg.get("cluster_labels", {})}

    us_vote = aggregate_cluster_vote(
        us_symbols,
        closes_map,
        threshold_pct=threshold,
        min_indices=min_idx,
        labels=label_bundle,
        region="us",
        min_move_pct=min_move,
    )
    eu_vote = aggregate_cluster_vote(
        eu_symbols,
        closes_map,
        threshold_pct=threshold,
        min_indices=min_idx,
        labels=label_bundle,
        region="eu",
        min_move_pct=min_move,
    )
    return macro_snapshot_from_votes(
        us_vote,
        eu_vote,
        macro_cfg,
        statarb_spreads=statarb_spreads,
        hmm_state=hmm_state,
        hmm_prob=hmm_prob,
    )


def apply_m5_fallback_to_snapshot(
    snapshot: MacroSnapshot,
    *,
    us_symbols: list[str],
    eu_symbols: list[str],
    fallback_closes: dict[str, list[float]],
    macro_cfg: dict[str, Any] | None = None,
) -> MacroSnapshot:
    """Reavalia cluster flat em M15 usando fechamentos M5 com limiar menor."""
    cfg = resolve_macro_config(macro_cfg)
    if not cfg["cluster_use_m5_fallback_when_flat"]:
        return snapshot
    threshold = float(cfg["cluster_return_threshold_pct"])
    min_idx = int(cfg["min_indices_for_vote"])
    min_move_fb = float(cfg["cluster_fallback_min_move_pct"])
    label_bundle = {"cluster_labels": cfg.get("cluster_labels", {})}

    us_vote = ClusterVote(snapshot.us_dir, snapshot.us_strength, snapshot.us_parts)
    eu_vote = ClusterVote(snapshot.eu_dir, snapshot.eu_strength, snapshot.eu_parts)
    suffix_parts: list[str] = []

    if snapshot.us_dir == "flat":
        fb_us = aggregate_cluster_vote(
            us_symbols,
            fallback_closes,
            threshold_pct=threshold,
            min_indices=min_idx,
            labels=label_bundle,
            region="us",
            min_move_pct=min_move_fb,
        )
        if fb_us.direction != "flat":
            us_vote = fb_us
            suffix_parts.append("CLUSTER_M5_FALLBACK_US")
    if snapshot.eu_dir == "flat":
        fb_eu = aggregate_cluster_vote(
            eu_symbols,
            fallback_closes,
            threshold_pct=threshold,
            min_indices=min_idx,
            labels=label_bundle,
            region="eu",
            min_move_pct=min_move_fb,
        )
        if fb_eu.direction != "flat":
            eu_vote = fb_eu
            suffix_parts.append("CLUSTER_M5_FALLBACK_EU")

    if us_vote.direction == snapshot.us_dir and eu_vote.direction == snapshot.eu_dir:
        return snapshot
    return macro_snapshot_from_votes(
        us_vote,
        eu_vote,
        macro_cfg,
        cluster_suffix=" | ".join(suffix_parts),
        statarb_spreads=snapshot.statarb_spreads,
        hmm_state=snapshot.hmm_state,
        hmm_prob=snapshot.hmm_prob,
    )
