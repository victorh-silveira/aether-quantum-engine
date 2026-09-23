"""Inferencia estatistica para promocao de modelos binarios."""

import math


_Z_BY_CONFIDENCE = {0.90: 1.644854, 0.95: 1.959964}


def wilson_lower_bound(*, wins: int, trials: int, confidence: float) -> float:
    """Calcula o limite inferior de Wilson para uma proporcao binaria."""
    n = int(trials)
    if n <= 0:
        return 0.0
    z = _Z_BY_CONFIDENCE.get(round(float(confidence), 2))
    if z is None:
        raise ValueError("settlement_confidence deve ser 0.90 ou 0.95")
    p = min(1.0, max(0.0, float(wins) / float(n)))
    z2 = z * z
    center = p + z2 / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return max(0.0, (center - spread) / (1.0 + z2 / n))
