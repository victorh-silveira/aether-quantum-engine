"""Expansao assimétrica da fracao Kelly em super-consenso estatistico."""

from __future__ import annotations

from typing import Any

from src.domain.risk.kelly_base_fraction import resolve_effective_kelly_fraction


def _order_side(order_direction: str | None) -> str | None:
    """Normaliza direcao da ordem para CALL ou PUT."""
    side = str(order_direction or "").strip().upper()
    if side in ("CALL", "PUT"):
        return side
    return None


def _call_side_probabilities(metrics: dict) -> list[float]:
    """Coleta probabilidades de CALL disponiveis nas metricas."""
    probs: list[float] = []
    calibrated = metrics.get("calibrated_prob")
    raw = metrics.get("raw_prob")
    if calibrated is not None:
        probs.append(float(calibrated))
    if raw is not None:
        probs.append(float(raw))
    return probs


def resolve_order_side_probability(metrics: dict, order_direction: str | None) -> float:
    """Retorna probabilidade alinhada a ordem a partir de calibrated_prob ou raw_prob."""
    probs = _call_side_probabilities(metrics)
    if not probs:
        return 0.0
    side = _order_side(order_direction)
    if side == "PUT":
        return 1.0 - min(probs)
    return max(probs)


def is_unanimous_vote_alignment(
    call_votes: int,
    put_votes: int,
    order_direction: str | None,
) -> bool:
    """True quando votos tecnicos sao 6x0 ou 0x6 alinhados a ordem."""
    side = _order_side(order_direction)
    cv = int(call_votes)
    pv = int(put_votes)
    if side == "CALL":
        return cv >= 6 and pv == 0
    if side == "PUT":
        return pv >= 6 and cv == 0
    return False


def resolve_super_concordance_fraction_multiplier(
    metrics: dict | None,
    order_direction: str | None,
    kelly_config: dict[str, Any],
    *,
    recovery_active: bool,
) -> float:
    """Retorna multiplicador da fracao Kelly base (1.0 ou super_concordance_booster)."""
    cfg = kelly_config if isinstance(kelly_config, dict) else {}
    if recovery_active or not bool(cfg.get("super_concordance_enabled", True)):
        return 1.0
    if not isinstance(metrics, dict):
        return 1.0
    indicators = metrics.get("indicators")
    ind = indicators if isinstance(indicators, dict) else {}
    hurst_min = float(cfg.get("super_concordance_hurst_min", 0.55))
    if float(ind.get("hurst", 0.0)) <= hurst_min:
        return 1.0
    if not is_unanimous_vote_alignment(
        int(metrics.get("call_votes", 0)),
        int(metrics.get("put_votes", 0)),
        order_direction,
    ):
        return 1.0
    prob_min = float(cfg.get("super_concordance_prob_min", 0.75))
    if resolve_order_side_probability(metrics, order_direction) + 1e-9 < prob_min:
        return 1.0
    return float(cfg.get("super_concordance_booster", 1.5))


def apply_super_concordance_kelly_fraction(
    kelly_f: float,
    kelly_config: dict[str, Any],
    dl_metrics: dict | None,
    order_direction: str | None,
    *,
    recovery_active: bool,
) -> float:
    """Calcula f* com fracao Kelly expandida quando super-consenso estiver ativo."""
    base_fraction = resolve_effective_kelly_fraction(kelly_config, recovery_active=recovery_active)
    multiplier = resolve_super_concordance_fraction_multiplier(
        dl_metrics,
        order_direction,
        kelly_config,
        recovery_active=recovery_active,
    )
    effective_fraction = base_fraction * multiplier
    if isinstance(dl_metrics, dict) and multiplier > 1.0:
        dl_metrics["super_concordance_booster_active"] = True
        dl_metrics["super_concordance_fraction_multiplier"] = multiplier
        dl_metrics["kelly_fraction_effective"] = effective_fraction
    return max(0.0, float(kelly_f) * effective_fraction)
