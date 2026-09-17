"""Temperatura de export para recuperar nitidez OOS do TCN."""

from __future__ import annotations

import logging

from src.application.services.deep_learning.dl_calibration import (
    _METHOD_TEMPERATURE,
    CalibratorState,
    apply_calibrator_stable,
    temperature_bounds,
)
from src.application.services.deep_learning.dl_sharpness import mean_sharpness


logger = logging.getLogger("AETH")


def maybe_temperature_sharpen_for_export(
    preferred: CalibratorState,
    *,
    val_probs: list[float],
    min_oos_sharpness: float,
) -> tuple[CalibratorState, float]:
    """Se OOS ainda fica abaixo do piso, tenta temperatura T in [temp_min, 1] para nitidez."""
    if not val_probs:
        return preferred, 0.0
    floor = float(min_oos_sharpness)
    temp_min, _temp_max = temperature_bounds()
    best = preferred
    best_sharp = mean_sharpness([float(apply_calibrator_stable(float(p), preferred)) for p in val_probs])
    if best_sharp + 1e-3 >= floor:
        return preferred, best_sharp
    steps = 11
    for idx in range(steps):
        frac = idx / max(steps - 1, 1)
        temp = 1.0 - frac * (1.0 - float(temp_min))
        cal = CalibratorState(method=_METHOD_TEMPERATURE, temperature=float(temp), platt_a=1.0, platt_b=0.0)
        sharp = mean_sharpness([float(apply_calibrator_stable(float(p), cal)) for p in val_probs])
        if sharp > best_sharp + 1e-12:
            best, best_sharp = cal, sharp
        if sharp + 1e-3 >= floor:
            logger.info(
                "DL_CAL: temperature sharpen T=%.3f sharpness=%.4f >= min=%.4f",
                temp,
                sharp,
                floor,
            )
            return cal, sharp
    return best, best_sharp
