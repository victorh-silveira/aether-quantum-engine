"""Blend de probabilidade TCN com consenso de indicadores tecnicos."""

from __future__ import annotations


def indicator_vote_share(call_votes: int, put_votes: int) -> tuple[float, float, int]:
    """Retorna (call_share, put_share, total) normalizados."""
    total = max(0, int(call_votes)) + max(0, int(put_votes))
    if total <= 0:
        return 0.5, 0.5, 0
    return float(call_votes) / float(total), float(put_votes) / float(total), total


def blend_prob_with_indicator_consensus(
    prob: float,
    call_votes: int,
    put_votes: int,
    *,
    adx: float | None = None,
    adx_min: float = 0.12,
    min_votes: int = 4,
    majority_ratio: float = 0.65,
    max_boost: float = 0.08,
    gray_half_width: float = 0.04,
) -> tuple[float, float, str]:
    """Desloca probabilidade neutra do TCN quando o consenso tecnico e forte."""
    p = max(0.0, min(1.0, float(prob)))
    if abs(p - 0.5) > float(gray_half_width) + 1e-12:
        return p, 0.0, "tcn_decisive"
    if adx is not None and float(adx) + 1e-12 < float(adx_min):
        return p, 0.0, "adx_weak"
    call_share, put_share, total = indicator_vote_share(call_votes, put_votes)
    if total < int(min_votes):
        return p, 0.0, "votes_low"
    boost_cap = max(0.0, float(max_boost))
    if call_share + 1e-12 >= float(majority_ratio):
        strength = max(0.0, min(1.0, (call_share - 0.5) * 2.0))
        delta = boost_cap * strength
        return min(1.0, p + delta), delta, "call_consensus"
    if put_share + 1e-12 >= float(majority_ratio):
        strength = max(0.0, min(1.0, (put_share - 0.5) * 2.0))
        delta = boost_cap * strength
        return max(0.0, p - delta), -delta, "put_consensus"
    return p, 0.0, "no_majority"
