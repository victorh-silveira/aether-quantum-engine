"""Guardrails de execute para simbolos propagados do cluster."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_post_loss import cluster_post_loss_block_reason
from src.application.services.llm.cluster_statarb_select import (
    resolve_statarb_cluster_config_for_tag,
    symbol_z_supports_direction,
)
from src.application.services.llm.macro_config import resolve_macro_config
from src.application.services.llm.profitable_scenario import (
    cluster_symbol_allowed_for_tag,
    min_conviction_for_macro_tag,
)
from src.domain.models.trade import TradeDirection


def min_conviction_execute(orch: Any) -> float:
    """Le piso de conviccao para entrada a partir de llm e risk_management."""
    llm_cfg = orch.config.get("llm", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    rm = orch.config.get("risk_management", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    limits = rm.get("limits", {}) if isinstance(rm.get("limits"), dict) else {}
    value = max(0.0, min(0.99, float(llm_cfg.get("min_conviction_execute", 0.80))))
    mc_lim = limits.get("min_conviction_execute")
    if mc_lim is not None:
        value = max(value, float(mc_lim))
    return value


def cluster_tag_region_ok(
    tag: str,
    us_s: float,
    eu_s: float,
    floor: float,
    cfg: dict[str, Any],
    active_region: str | None,
) -> bool:
    """Valida forca regional minima para a tag macro ativa."""
    div_floor = float(cfg.get("divergence_min_leader_strength", floor + 0.05))
    checks = {
        "risk_on": us_s >= floor,
        "risk_off": eu_s >= floor,
        "divergence_us_leads": us_s >= div_floor,
        "divergence_eu_leads": eu_s >= div_floor,
    }
    if tag in checks:
        return checks[tag]
    if tag not in ("", "indefinido"):
        return False
    indef_floor = float(cfg.get("indefinido_min_leader_strength", floor + 0.03))
    if active_region == "us":
        return us_s >= indef_floor
    if active_region == "eu":
        return eu_s >= indef_floor
    return max(us_s, eu_s) >= indef_floor


def cluster_entry_allowed(
    metrics: dict[str, Any],
    macro_cfg: dict[str, Any] | None,
    *,
    active_region: str | None,
    llm_cluster_explicit: bool = False,
) -> bool:
    """Valida tag, HMM e forca regional do cluster ativo sem herdar execute da ancora."""
    cfg = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
    tag = str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")
    allowed_tags = cfg.get("allowed_execute_tags")
    if allowed_tags and tag not in allowed_tags:
        return False
    hmm_prob = float(metrics.get("hmm_prob", 1.0))
    if hmm_prob < float(cfg.get("assert_min_hmm_prob", 0.0)):
        return False
    if llm_cluster_explicit:
        floor = float(cfg["confluence_conviction_floor"])
        us_s = float(metrics.get("macro_us_strength_quant", 0))
        eu_s = float(metrics.get("macro_eu_strength_quant", 0))
        if tag in ("risk_on", "risk_off", "divergence_us_leads", "divergence_eu_leads"):
            return cluster_tag_region_ok(tag, us_s, eu_s, floor, cfg, active_region)
        return True
    floor = float(cfg["confluence_conviction_floor"])
    us_s = float(metrics.get("macro_us_strength_quant", 0))
    eu_s = float(metrics.get("macro_eu_strength_quant", 0))
    return cluster_tag_region_ok(tag, us_s, eu_s, floor, cfg, active_region)


def cluster_execute_flag(
    orch: Any,
    metrics: dict[str, Any],
    conviction: float,
    target_direction: TradeDirection | None,
    macro_cfg: dict[str, Any] | None,
    corr_cfg: dict[str, Any] | None,
    *,
    active_region: str | None,
    target_sym: str,
    llm_cluster_explicit: bool = False,
) -> bool:
    """Recalcula execute nos indices; nao propaga veto macro da ancora FX."""
    return (
        cluster_execute_block_reason(
            orch,
            metrics,
            conviction,
            target_direction,
            macro_cfg,
            corr_cfg,
            active_region=active_region,
            target_sym=target_sym,
            llm_cluster_explicit=llm_cluster_explicit,
        )
        == "allowed"
    )


def cluster_execute_block_reason(
    orch: Any,
    metrics: dict[str, Any],
    conviction: float,
    target_direction: TradeDirection | None,
    macro_cfg: dict[str, Any] | None,
    corr_cfg: dict[str, Any] | None,
    *,
    active_region: str | None,
    target_sym: str,
    llm_cluster_explicit: bool = False,
    index_note: str = "",
) -> str:
    """Retorna motivo de bloqueio de execute para decisao de cluster."""
    _ = index_note
    reason = "allowed"
    post_loss = cluster_post_loss_block_reason(orch, target_sym=target_sym, target_direction=target_direction)
    if post_loss is not None:
        reason = post_loss
    elif target_direction is None:
        reason = "no_direction"
    else:
        macro_tag = str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")
        conv_floor = min_conviction_for_macro_tag(
            macro_cfg if isinstance(macro_cfg, dict) else None,
            macro_tag=macro_tag,
            base_floor=min_conviction_execute(orch),
        )
        if conviction < conv_floor:
            reason = "low_conviction"
        elif not cluster_symbol_allowed_for_tag(
            macro_cfg if isinstance(macro_cfg, dict) else None,
            macro_tag=macro_tag,
            symbol=target_sym,
        ):
            reason = "scenario_symbol_not_allowed"
    if reason == "allowed" and not cluster_entry_allowed(
        metrics,
        macro_cfg,
        active_region=active_region,
        llm_cluster_explicit=llm_cluster_explicit,
    ):
        reason = "macro_or_hmm_veto"
    if reason == "allowed" and not llm_cluster_explicit:
        c = corr_cfg if isinstance(corr_cfg, dict) else {}
        macro_tag = str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")
        spreads = metrics.get("statarb_spreads")
        if bool(c.get("statarb_require_z_align", True)) and isinstance(spreads, dict) and target_sym in spreads:
            statarb_cfg = resolve_statarb_cluster_config_for_tag(
                c,
                macro_cfg if isinstance(macro_cfg, dict) else None,
                macro_tag,
            )
            z = float(spreads[target_sym])
            hmm_state = int(metrics.get("hmm_state", 0))
            z_threshold = float(statarb_cfg.get("z_threshold", 2.5))
            min_abs_gate = float(statarb_cfg.get("min_abs_z", 0.0))
            if not symbol_z_supports_direction(
                z,
                target_direction,
                hmm_state=hmm_state,
                z_threshold=z_threshold,
                min_abs_z=min_abs_gate,
            ):
                reason = "statarb_z_misaligned"
    return reason
