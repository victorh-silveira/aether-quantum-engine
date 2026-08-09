"""Adaptacao de lado e sync de Kelly pela fita multi-escala (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_adapt_majority import adapt_on_majority_votes
from src.application.services.execution_scale_adapt_regimes import (
    adapt_on_retraction as _adapt_on_retraction,
    try_regime_adapts,
)
from src.application.services.execution_scale_micro import classify_micro_regime
from src.application.services.execution_scale_tape import mini_bar_pair_agrees
from src.application.services.execution_scale_vision import parse_scale_vision_config
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import apply_kelly_side_p
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


__all__ = (
    "apply_scale_direction_adapt",
    "apply_scale_kelly_side_sync",
    "_adapt_on_retraction",
)


def _calibrated_side(metrics: dict[str, Any]) -> TradeDirection | None:
    """Lado implicito por calibrated_prob (P(CALL)); None se ausente/invalido."""
    if "calibrated_prob" not in metrics or metrics.get("calibrated_prob") is None:
        return None
    try:
        cal = float(metrics["calibrated_prob"])
    except (TypeError, ValueError):
        return None
    if cal + 1e-12 >= 0.5:
        return TradeDirection.CALL
    return TradeDirection.PUT


def _accept_adapted(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    adapted: TradeDirection,
    *,
    reason: str,
    require_cal_agree: bool,
) -> TradeDirection:
    """Aplica adaptacao se Cal concorda (quando exigido); senao mantem TCN."""
    if adapted == exec_dir:
        return exec_dir
    if require_cal_agree:
        cal_side = _calibrated_side(metrics)
        if cal_side is not None and cal_side != adapted:
            metrics["scale_adapted"] = False
            metrics["scale_adapt_reason"] = "cal_disagree"
            metrics["scale_adapt_blocked_side"] = adapted.name
            return exec_dir
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = reason
    return adapted


def apply_scale_direction_adapt(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Adapta exec_direction ao consenso da fita, maioria de votos ou regimes micro."""
    cfg = parse_scale_vision_config(None)
    metrics["tcn_direction"] = exec_dir.name
    metrics.setdefault("scale_adapted", False)
    metrics.setdefault("scale_adapt_reason", "idle")
    classify_micro_regime(metrics, exec_dir.name, cfg=cfg)
    if not bool(cfg.get("enabled", True)):
        metrics["scale_adapt_reason"] = "disabled"
        return exec_dir
    if not bool(cfg.get("adapt_direction_enabled", True)):
        metrics["scale_adapt_reason"] = "adapt_off"
        return exec_dir
    require_cal = bool(cfg.get("adapt_require_cal_agree", True))
    skip_chop = bool(cfg.get("adapt_skip_chop", True))
    if skip_chop and str(metrics.get("scale_micro_regime") or "").lower() == "chop":
        metrics["scale_adapt_reason"] = "chop_hold"
        return exec_dir
    majority = adapt_on_majority_votes(metrics, exec_dir, cfg)
    if majority is not None:
        reason = str(metrics.get("scale_adapt_reason") or "majority_votes")
        return _accept_adapted(metrics, exec_dir, majority, reason=reason, require_cal_agree=require_cal)
    consensus = str(metrics.get("scale_tape_consensus") or "").upper()
    if consensus not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        regime = try_regime_adapts(metrics, exec_dir, cfg)
        if regime is not None:
            reason = str(metrics.get("scale_adapt_reason") or "regime")
            return _accept_adapted(metrics, exec_dir, regime, reason=reason, require_cal_agree=require_cal)
        metrics["scale_adapt_reason"] = "no_consensus"
        return exec_dir
    if consensus == exec_dir.name:
        metrics["scale_adapt_reason"] = "aligned"
        return exec_dir
    if bool(cfg.get("adapt_require_bar_pair_agree", True)) and not mini_bar_pair_agrees(metrics, consensus):
        regime = try_regime_adapts(metrics, exec_dir, cfg)
        if regime is not None:
            reason = str(metrics.get("scale_adapt_reason") or "regime")
            return _accept_adapted(metrics, exec_dir, regime, reason=reason, require_cal_agree=require_cal)
        metrics["scale_adapt_reason"] = "need_bar_pair"
        return exec_dir
    mode = str(metrics.get("calibration_mode") or "")
    raw_ok = mode == "raw_extreme"
    strong_ok = bool(cfg.get("adapt_allow_strong_tape", False)) and bool(metrics.get("scale_tape_strong"))
    if raw_ok or strong_ok:
        adapted = TradeDirection[consensus]
        reason = "tape_strong" if strong_ok and not raw_ok else "tape_vs_tcn"
        return _accept_adapted(metrics, exec_dir, adapted, reason=reason, require_cal_agree=require_cal)
    if bool(cfg.get("adapt_require_raw_extreme", True)):
        regime = try_regime_adapts(metrics, exec_dir, cfg)
        if regime is not None:
            reason = str(metrics.get("scale_adapt_reason") or "regime")
            return _accept_adapted(metrics, exec_dir, regime, reason=reason, require_cal_agree=require_cal)
        metrics["scale_adapt_reason"] = "need_raw_extreme"
        return exec_dir
    adapted = TradeDirection[consensus]
    return _accept_adapted(metrics, exec_dir, adapted, reason="tape_vs_tcn", require_cal_agree=require_cal)


def apply_scale_kelly_side_sync(metrics: dict[str, Any], exec_dir: TradeDirection) -> dict[str, Any]:
    """Alinha p Kelly ao lado executado com piso SSOT em todo candidato."""
    rt = load_kelly_runtime_from_settings()
    conviction = float(metrics.get("conviction", metrics.get("trade_score", 0.5)) or 0.5)
    apply_kelly_side_p(
        metrics,
        order_direction=exec_dir.name,
        kelly_config={"kelly_p_floor": rt["kelly_p_floor"]},
        conviction=conviction,
    )
    return metrics
