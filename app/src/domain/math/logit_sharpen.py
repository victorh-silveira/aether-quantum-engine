"""Escala termica e sharpening nos logits na camada de dominio matematico."""

from __future__ import annotations

import math


def logit(prob: float) -> float:
    """Converte probabilidade p em logit de forma estavel."""
    p = min(max(float(prob), 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    """Converte logit z de volta para probabilidade sigmoide."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def sharpen_logit_temperature(prob: float, tau: float = 0.40) -> float:
    """Aplica sharpening termico dividindo o logit por tau <= 0.40."""
    t = max(0.10, min(float(tau), 0.99))
    z = logit(prob)
    return sigmoid(z / t)


def resolve_sharpening_tau(
    base_tau: float = 0.40,
    *,
    vol_ratio: float = 1.0,
    min_tau: float = 0.15,
    max_tau: float = 0.40,
) -> float:
    """Modula a temperatura de sharpening inversamente com base na volatilidade."""
    ratio = max(0.2, min(float(vol_ratio), 5.0))
    adjusted = float(base_tau) / math.sqrt(ratio)
    return max(float(min_tau), min(float(max_tau), adjusted))


def apply_dynamic_temperature_sharpening(
    prob: float,
    *,
    base_tau: float = 0.40,
    margin_threshold: float = 0.035,
    vol_ratio: float = 1.0,
    force: bool = False,
) -> tuple[float, bool]:
    """Aplica expansao termica nos logits se a probabilidade estiver colapsada em torno de 0.50."""
    p = float(prob)
    margin = abs(p - 0.5)
    if not force and margin >= float(margin_threshold):
        return p, False
    tau_eff = resolve_sharpening_tau(base_tau, vol_ratio=vol_ratio)
    sharpened = sharpen_logit_temperature(p, tau_eff)
    return sharpened, True
