"""Adaptacao de lado e sync de Kelly pela fita multi-escala (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_micro import classify_micro_regime
from src.application.services.execution_scale_tape import mini_bar_pair_agrees
from src.application.services.execution_scale_vision import parse_scale_vision_config
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import apply_kelly_side_p
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


def _adapt_on_retraction(
    metrics: dict[str, Any], exec_dir: TradeDirection, cfg: dict[str, Any]
) -> TradeDirection | None:
    """Adapta ao lado vivo quando ha retracao confirmada contra o TCN."""
    if not bool(cfg.get("adapt_on_retraction", True)):
        return None
    classify_micro_regime(metrics, exec_dir.name, cfg=cfg)
    if not bool(metrics.get("scale_retraction_vs_tcn")):
        return None
    live = str(metrics.get("scale_micro_side") or "").upper()
    if live not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return None
    if live == exec_dir.name:
        return None
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = "retraction"
    return TradeDirection[live]


def apply_scale_direction_adapt(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Adapta exec_direction ao consenso da fita ou a retracao micro (sem SKIP)."""
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
    consensus = str(metrics.get("scale_tape_consensus") or "").upper()
    if consensus not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        retracted = _adapt_on_retraction(metrics, exec_dir, cfg)
        if retracted is not None:
            return retracted
        metrics["scale_adapt_reason"] = "no_consensus"
        return exec_dir
    if consensus == exec_dir.name:
        metrics["scale_adapt_reason"] = "aligned"
        return exec_dir
    if bool(cfg.get("adapt_require_bar_pair_agree", True)) and not mini_bar_pair_agrees(metrics, consensus):
        retracted = _adapt_on_retraction(metrics, exec_dir, cfg)
        if retracted is not None:
            return retracted
        metrics["scale_adapt_reason"] = "need_bar_pair"
        return exec_dir
    mode = str(metrics.get("calibration_mode") or "")
    raw_ok = mode == "raw_extreme"
    strong_ok = bool(cfg.get("adapt_allow_strong_tape", True)) and bool(metrics.get("scale_tape_strong"))
    if raw_ok or strong_ok:
        adapted = TradeDirection[consensus]
        metrics["scale_adapted"] = True
        metrics["scale_adapt_reason"] = "tape_strong" if strong_ok and not raw_ok else "tape_vs_tcn"
        return adapted
    if bool(cfg.get("adapt_require_raw_extreme", True)):
        retracted = _adapt_on_retraction(metrics, exec_dir, cfg)
        if retracted is not None:
            return retracted
        metrics["scale_adapt_reason"] = "need_raw_extreme"
        return exec_dir
    adapted = TradeDirection[consensus]
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = "tape_vs_tcn"
    return adapted


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
