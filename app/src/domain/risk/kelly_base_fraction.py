"""Compressao da fracao Kelly base fora de recovery."""

from typing import Any


KELLY_FRACTION_BASE_RETENTION = 0.40
KELLY_FRACTION_REFERENCE = 0.0035
KELLY_FRACTION_COMPRESSED = 0.0012


def _compress_kelly_base_fraction(base_fraction: float) -> float:
    """Comprime fracao Kelly em 60% com referencia 0.0035 -> 0.0012."""
    if base_fraction <= 0.0:
        return 0.0
    if abs(base_fraction - KELLY_FRACTION_REFERENCE) < 1e-12:
        return KELLY_FRACTION_COMPRESSED
    return base_fraction * KELLY_FRACTION_BASE_RETENTION


def resolve_effective_kelly_fraction(
    kelly_config: dict[str, Any],
    *,
    recovery_active: bool = False,
) -> float:
    """Retorna fracao Kelly base com corte linear de 60% em regime normal."""
    base_fraction = float(kelly_config.get("fraction", 0.03))
    if recovery_active:
        return base_fraction
    explicit_scale = kelly_config.get("kelly_fraction_scale")
    if explicit_scale is not None:
        return base_fraction * max(0.0, float(explicit_scale))
    return _compress_kelly_base_fraction(base_fraction)
