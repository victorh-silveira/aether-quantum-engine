"""Adaptacao de lado pela fita multi-escala (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_vision import parse_scale_vision_config
from src.domain.models.trade import TradeDirection


def apply_scale_direction_adapt(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Adapta exec_direction ao consenso da fita quando opoe o TCN sob raw_extreme."""
    cfg = parse_scale_vision_config(None)
    metrics["tcn_direction"] = exec_dir.name
    metrics.setdefault("scale_adapted", False)
    metrics.setdefault("scale_adapt_reason", "idle")
    if not bool(cfg.get("enabled", True)):
        metrics["scale_adapt_reason"] = "disabled"
        return exec_dir
    if not bool(cfg.get("adapt_direction_enabled", True)):
        metrics["scale_adapt_reason"] = "adapt_off"
        return exec_dir
    consensus = str(metrics.get("scale_tape_consensus") or "").upper()
    if consensus not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        metrics["scale_adapt_reason"] = "no_consensus"
        return exec_dir
    if consensus == exec_dir.name:
        metrics["scale_adapt_reason"] = "aligned"
        return exec_dir
    if bool(cfg.get("adapt_require_raw_extreme", True)):
        mode = str(metrics.get("calibration_mode") or "")
        if mode != "raw_extreme":
            metrics["scale_adapt_reason"] = "need_raw_extreme"
            return exec_dir
    adapted = TradeDirection[consensus]
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = "tape_vs_tcn"
    return adapted
