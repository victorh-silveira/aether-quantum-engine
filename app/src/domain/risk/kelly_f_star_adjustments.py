"""Ajustes de f* Kelly: consenso, escala defensiva e piso por divergencia."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_penalty import consensus_kelly_retention
from src.domain.risk.stake_sizing import clamp_kelly_stake, consensus_entropy_applies_min_stake


def apply_kelly_fraction_scale(f_star: float, dl_metrics: dict | None) -> float:
    """Atenua fracao Kelly quando resolver sinaliza execucao defensiva."""
    if not isinstance(dl_metrics, dict):
        return f_star
    frac_scale = float(dl_metrics.get("kelly_fraction_scale", 1.0))
    if frac_scale < 1.0:
        return f_star * max(0.0, frac_scale)
    return f_star


def apply_consensus_entropy_f_star(
    rm: Any,
    f_star: float,
    dl_metrics: dict | None,
    order_direction: str | None,
    *,
    silent: bool,
) -> float:
    """Atenua f* quando ordem diverge do consenso tecnico."""
    if not isinstance(dl_metrics, dict):
        return f_star
    retention = consensus_kelly_retention(
        dl_metrics,
        order_direction,
        kelly_config=rm.kelly_config,
        consecutive_losses=int(getattr(rm, "consecutive_losses_linear", 0)),
        pending_loss_total=sum(getattr(rm, "pending_loss", {}).values()),
    )
    if retention < 1.0 and not silent:
        rm.logger.debug(
            "KELLY: consensus retention=%.2f ord=%s votes=%d/%d",
            retention,
            order_direction,
            int(dl_metrics.get("call_votes", 0)),
            int(dl_metrics.get("put_votes", 0)),
        )
    dl_metrics["consensus_entropy_retention"] = retention
    return f_star * retention


def kelly_base_with_consensus_floor(
    bankroll: float,
    f_star: float,
    dl_metrics: dict | None,
    kelly_config: dict,
    sizing_conviction: float,
    stake_min: float,
) -> float:
    """Calcula kelly_base e cancela (retorna 0.0) quando f_star <= 0.0."""
    if f_star <= 0.0:
        return 0.0
    kelly_base = clamp_kelly_stake(bankroll, bankroll * f_star, kelly_config, sizing_conviction)
    if isinstance(dl_metrics, dict) and consensus_entropy_applies_min_stake(
        float(dl_metrics.get("consensus_entropy_retention", 1.0)),
        kelly_config,
    ):
        return stake_min
    return kelly_base
