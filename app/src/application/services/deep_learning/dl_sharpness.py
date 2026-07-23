"""Metricas e gates de nitidez (sharpness) de probabilidades TCN."""

from __future__ import annotations

from typing import Any


def probability_margin(prob: float) -> float:
    """Distancia absoluta da probabilidade ao neutro 0.5."""
    return abs(float(prob) - 0.5)


def mean_sharpness(probs: list[float]) -> float:
    """Media de |p-0.5| sobre uma amostra de probabilidades."""
    if not probs:
        return 0.0
    total = 0.0
    for prob in probs:
        total += probability_margin(prob)
    return total / float(len(probs))


def sharpness_pass_fraction(probs: list[float], *, floor: float) -> float:
    """Fracao de amostras com margem >= floor."""
    if not probs:
        return 0.0
    threshold = float(floor)
    hits = sum(1 for prob in probs if probability_margin(prob) + 1e-12 >= threshold)
    return float(hits) / float(len(probs))


def assert_export_sharpness_value(
    sharpness: float,
    *,
    floor: float,
    label: str = "OOS",
) -> float:
    """Bloqueia export quando a sharpness media ja calculada fica abaixo do piso."""
    value = float(sharpness)
    min_floor = float(floor)
    if value + 1e-12 < min_floor:
        raise RuntimeError(f"Export TCN bloqueado: sharpness {label}={value:.4f} < min={min_floor:.4f}")
    return value


def assert_export_sharpness_floor(
    probs: list[float],
    *,
    floor: float,
    label: str = "OOS",
) -> float:
    """Bloqueia export de modelo com sharpness media abaixo do piso quality-first."""
    return assert_export_sharpness_value(mean_sharpness(probs), floor=floor, label=label)


def resolve_calibration_sharpness_cfg(calibration_cfg: dict[str, Any] | None) -> dict[str, float]:
    """Resolve pisos de sharpness/margem a partir do bloco calibration."""
    cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    return {
        "min_calibration_sharpness": float(cfg.get("min_calibration_sharpness", 0.03)),
        "min_calibration_margin_floor": float(cfg.get("min_calibration_margin_floor", 0.03)),
        "min_oos_sharpness": float(cfg.get("min_oos_sharpness", 0.03)),
    }
