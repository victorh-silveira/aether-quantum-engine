"""Stake do pullback com base de 1% e recuperacao pelo payout cotado."""

from __future__ import annotations

import math


def pullback_stake(
    available: float,
    pending: float,
    payout_rate: float,
    stake_min: float,
    single_strike_target: float = 0.0,
    amort_steps: int = 1,
) -> tuple[float, float]:
    """Devolve stake e parcela da perda que uma vitoria poderia amortizar."""
    if not all(math.isfinite(value) for value in (available, pending, payout_rate, stake_min, single_strike_target)):
        return 0.0, 0.0
    if available <= 0 or payout_rate <= 0 or stake_min <= 0:
        return 0.0, 0.0
    cap = math.floor(min(available, available * 0.05) * 100) / 100
    if cap + 1e-9 < stake_min:
        return 0.0, 0.0
    base = math.floor(available * 0.01 * 100) / 100
    if pending > 0:
        steps = max(1, int(amort_steps))
        target = max(base, stake_min, (pending / steps) / payout_rate)
    elif single_strike_target > 0:
        target = max(base, stake_min, single_strike_target / payout_rate)
    else:
        target = max(base, stake_min)
    stake = min(cap, math.ceil(target * 100 - 1e-9) / 100)
    covered = min(max(0.0, pending), stake * payout_rate)
    return stake, covered
