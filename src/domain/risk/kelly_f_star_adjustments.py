"""Ajustes de f* Kelly: consenso, escala defensiva e clamp de banca."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.domain.analytics.sample_size_policy import explore_stake_scale
from src.domain.risk.consensus_stake_penalty import consensus_kelly_retention
from src.domain.risk.stake_sizing import clamp_kelly_stake
from src.domain.risk.super_concordance_kelly import resolve_super_concordance_kelly_fraction


def resolve_kelly_fraction_scale(
    f_star: float,
    dl_metrics: Any | None,
    *,
    stake_regime_override: str | None = None,
    kelly_fraction_scale_override: float | None = None,
) -> tuple[float, float | None]:
    """Atenua fracao Kelly e retorna (f_star_ajustado, explore_stake_scale) sem mutar entrada."""
    if dl_metrics is None or not hasattr(dl_metrics, "get"):
        return f_star, None
    out = float(f_star)
    frac_scale = (
        float(kelly_fraction_scale_override)
        if kelly_fraction_scale_override is not None
        else float(dl_metrics.get("kelly_fraction_scale", 1.0))
    )
    if frac_scale < 1.0:
        out = out * max(0.0, frac_scale)
    regime = str(stake_regime_override or dl_metrics.get("stake_regime") or "EXPLORE").upper()
    has_live_n = "live_n" in dl_metrics if hasattr(dl_metrics, "__contains__") else isinstance(dl_metrics, Mapping)
    if regime == "EXPLORE" and has_live_n:
        live_n = int(dl_metrics.get("live_n", 0) or 0)
        scale = float(dl_metrics.get("explore_stake_scale") or explore_stake_scale(live_n))
        return out * scale, scale
    return out, None


def apply_kelly_fraction_scale(f_star: float, dl_metrics: Any | None) -> float:
    """Atenua fracao Kelly quando resolver sinaliza execucao defensiva ou N pequeno."""
    out, _scale = resolve_kelly_fraction_scale(f_star, dl_metrics)
    return out


def resolve_consensus_entropy_f_star(
    rm: Any,
    f_star: float,
    dl_metrics: Any | None,
    order_direction: str | None,
    *,
    silent: bool,
) -> tuple[float, float]:
    """Atenua f* quando ordem diverge do consenso tecnico e retorna (f_star, retention)."""
    if dl_metrics is None or not hasattr(dl_metrics, "get"):
        return f_star, 1.0
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
    return f_star * retention, retention


def apply_consensus_entropy_f_star(
    rm: Any,
    f_star: float,
    dl_metrics: Any | None,
    order_direction: str | None,
    *,
    silent: bool,
) -> float:
    """Atenua f* quando ordem diverge do consenso tecnico."""
    adjusted, _retention = resolve_consensus_entropy_f_star(rm, f_star, dl_metrics, order_direction, silent=silent)
    return adjusted


def kelly_base_with_consensus_floor(
    bankroll: float,
    f_star: float,
    dl_metrics: Any | None,
    kelly_config: dict,
    sizing_conviction: float,
    stake_min: float = 1.0,
) -> float:
    """Calcula kelly_base via clamp percentual; stake_min nao inventa sizing."""
    _ = (dl_metrics, stake_min)
    f = max(0.0, float(f_star))
    return clamp_kelly_stake(bankroll, bankroll * f, kelly_config, sizing_conviction)


def collect_f_star_telemetry(
    rm: Any,
    *,
    kelly_f: float,
    dl_metrics: Any | None,
    order_direction: Any,
    recovery_active: bool,
    recovery_bypass_consensus: bool,
    silent: bool,
    stake_regime_override: str | None = None,
    kelly_fraction_scale_override: float | None = None,
) -> dict[str, Any]:
    """Calcula metadados de f* (super-concordance, entropia, explore_scale) sem mutar entrada."""
    f_adj, mult, eff_frac = resolve_super_concordance_kelly_fraction(
        kelly_f, rm.kelly_config, dl_metrics, order_direction, recovery_active=recovery_active
    )
    retention = 1.0
    if not recovery_bypass_consensus:
        f_adj, retention = resolve_consensus_entropy_f_star(rm, f_adj, dl_metrics, order_direction, silent=silent)
    _f_final, explore_scale = resolve_kelly_fraction_scale(
        f_adj,
        dl_metrics,
        stake_regime_override=stake_regime_override,
        kelly_fraction_scale_override=kelly_fraction_scale_override,
    )
    return {
        "explore_stake_scale": explore_scale,
        "consensus_entropy_retention": retention,
        "super_concordance_booster_active": bool(mult > 1.0),
        "super_concordance_fraction_multiplier": mult if mult > 1.0 else None,
        "kelly_fraction_effective": eff_frac if mult > 1.0 else None,
    }
