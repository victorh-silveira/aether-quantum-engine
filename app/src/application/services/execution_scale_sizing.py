"""Sizing soft por discordancia multi-escala (sem veto de direcao)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_vision import parse_scale_vision_config
from src.domain.models.trade import TradeDirection


def _should_dampen(metrics: dict[str, Any]) -> bool:
    """True quando ha discord, adapt, retracao ou mili contra o TCN."""
    if bool(metrics.get("scale_discordance")) or bool(metrics.get("scale_adapted")):
        return True
    regime = str(metrics.get("scale_micro_regime") or "").lower()
    if regime == "retraction":
        return True
    return regime == "chop" and bool(metrics.get("scale_mili_oppose_tcn"))


def apply_scale_kelly_sizing(
    orch: Any | None,
    symbol: str | None,
    direction: TradeDirection,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Atenua Kelly e forca explore quando escalas discordam, adaptam ou retrai."""
    _ = (orch, symbol, direction)
    cfg = parse_scale_vision_config(None)
    metrics.setdefault("scale_force_explore", False)
    metrics.setdefault("scale_max_stake_pct", None)
    if not bool(cfg.get("enabled", True)):
        metrics["scale_sizing_reason"] = "disabled"
        return metrics
    if not _should_dampen(metrics):
        metrics["scale_sizing_reason"] = "aligned"
        metrics["scale_kelly_mult"] = 1.0
        return metrics
    mult = float(cfg.get("kelly_mult_discord", 0.35))
    scale = float(metrics.get("kelly_fraction_scale", 1.0))
    metrics["kelly_fraction_scale"] = max(0.05, scale * mult)
    metrics["scale_kelly_mult"] = mult
    metrics["scale_max_stake_pct"] = float(cfg.get("max_stake_pct_discord", 0.005))
    if bool(cfg.get("block_recover_on_discord", True)):
        metrics["scale_force_explore"] = True
        reason = (
            "retraction" if str(metrics.get("scale_micro_regime") or "") == "retraction" else "discord_block_recover"
        )
        if bool(metrics.get("scale_adapted")):
            reason = str(metrics.get("scale_adapt_reason") or "adapt_block_recover")
        metrics["scale_sizing_reason"] = reason
    else:
        metrics["scale_sizing_reason"] = "discord_dampen"
    return metrics
