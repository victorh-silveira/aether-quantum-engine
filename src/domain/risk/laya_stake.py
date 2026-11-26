"""Dimensionamento dinamico de stake calibrado pela probabilidade do Laya (1% a 5%)."""

from __future__ import annotations

import math


def calculate_laya_stake(
    available_balance: float,
    probability: float,
    stake_min: float = 1.0,
    min_stake_pct: float = 0.01,
    max_stake_pct: float = 0.05,
) -> tuple[float, float]:
    """Calcula stake dinamica linear entre 1% e 5% da banca sem recuperacao ou Martingale."""
    if not all(math.isfinite(val) for val in (available_balance, probability, stake_min, min_stake_pct, max_stake_pct)):
        return 0.0, 0.0
    if available_balance <= 0.0 or stake_min <= 0.0:
        return 0.0, 0.0
    low_pct = min(0.05, max(0.01, min_stake_pct))
    high_pct = min(0.05, max(low_pct, max_stake_pct))
    norm = min(1.0, max(0.0, (probability - 0.5) / 0.5))
    pct = low_pct + norm * (high_pct - low_pct)
    cap = math.floor(available_balance * 0.05 * 100.0) / 100.0
    if cap + 1e-9 < stake_min:
        return 0.0, 0.0
    raw = math.floor(available_balance * pct * 100.0) / 100.0
    stake = min(cap, max(stake_min, raw))
    return stake, pct
