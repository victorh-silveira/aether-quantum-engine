"""Catalogo de razoes SKIP do playbook senior de binarias."""

from __future__ import annotations


SENIOR_SKIP_REASONS: frozenset[str] = frozenset(
    {
        "cal_margin_floor",
        "adx_min",
        "adx_starvation",
        "indicator_discordance",
        "adverse_micro_path",
        "meta_negative_edge",
        "val_accuracy_gate",
        "vol_ratio_starvation",
        "rsi_trend_misalign",
        "hurst_noise",
        "hurst_missing",
        "kelly_no_edge",
        "neutral_clamp",
        "price_zone_tcn_conflict",
    }
)


def is_senior_skip_reason(reason: str | None) -> bool:
    """True quando gate_reason pertence ao catalogo SKIP do playbook senior."""
    if reason is None:
        return False
    return str(reason).strip() in SENIOR_SKIP_REASONS
