"""Entropia de Bernoulli e penalizacao por incerteza calibrada."""

from __future__ import annotations

import math


def binary_entropy(probability: float, *, base: str = "e") -> float:
    """Entropia H(p) para evento binario; base e (natural) ou 2."""
    p = max(1e-12, min(1.0 - 1e-12, float(probability)))
    h_nat = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    if str(base).strip() == "2":
        return h_nat / math.log(2.0)
    return h_nat


def entropy_penalty_factor(
    probability: float,
    *,
    ceiling: float,
    floor: float = 0.0,
) -> float:
    """Fator em [0, 1]: 0 quando entropia abaixo do piso; 1 quando acima do teto."""
    ent = binary_entropy(probability)
    hi = max(float(ceiling), float(floor) + 1e-9)
    lo = max(0.0, min(float(floor), hi))
    if ent <= lo:
        return 0.0
    if ent >= hi:
        return 1.0
    return (ent - lo) / (hi - lo)


def adaptive_entropy_ceiling(
    base_ceiling: float,
    regime_score: float,
    *,
    squeeze_tighten: float,
    entropy_floor: float = 0.0,
) -> float:
    """Reduz teto de entropia quando regime de volatilidade indica compressao."""
    base = max(float(entropy_floor) + 1e-9, float(base_ceiling))
    tighten = max(0.0, min(1.0, float(squeeze_tighten)))
    regime = max(0.0, min(1.0, float(regime_score)))
    effective = base * (1.0 - tighten * regime)
    return max(float(entropy_floor), min(base, effective))
