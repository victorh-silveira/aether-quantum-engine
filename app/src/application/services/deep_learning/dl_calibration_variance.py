"""Guarda de variancia do calibrador DL (colapso std → identity)."""

from __future__ import annotations

import logging

from src.application.services.deep_learning.dl_calibration import (
    _METHOD_IDENTITY,
    CalibratorState,
    apply_calibrator,
)


logger = logging.getLogger("AETH")

_CAL_VARIANCE_COLLAPSE_STD = 1e-3
_CAL_VARIANCE_MIN_STD_RATIO = 0.5


def _identity_calibrator() -> CalibratorState:
    """Monta calibrador identity (raw) para persistencia apos colapso de std."""
    return CalibratorState(method=_METHOD_IDENTITY, temperature=1.0, platt_a=1.0, platt_b=0.0)


def maybe_identity_on_variance_collapse(
    preferred: CalibratorState,
    *,
    probs: list[float],
) -> CalibratorState:
    """Se o calibrador esmaga std vs raw no holdout/val, persiste identity."""
    if preferred.method == _METHOD_IDENTITY or not probs:
        return preferred
    raw_vals = [float(p) for p in probs]
    cal_vals = [float(apply_calibrator(float(p), preferred)) for p in raw_vals]
    if len(raw_vals) < 2 or len(cal_vals) < 2:
        return preferred
    raw_mean = sum(raw_vals) / float(len(raw_vals))
    cal_mean = sum(cal_vals) / float(len(cal_vals))
    raw_var = sum((v - raw_mean) ** 2 for v in raw_vals) / float(len(raw_vals))
    cal_var = sum((v - cal_mean) ** 2 for v in cal_vals) / float(len(cal_vals))
    raw_std = raw_var**0.5
    cal_std = cal_var**0.5
    if cal_std + 1e-12 >= _CAL_VARIANCE_COLLAPSE_STD and cal_std + 1e-12 >= (
        _CAL_VARIANCE_MIN_STD_RATIO * max(raw_std, 1e-12)
    ):
        return preferred
    logger.info(
        "DL_CAL: calibrador %s colapsou variancia (raw_std=%.6f cal_std=%.6f); identity.",
        preferred.method,
        raw_std,
        cal_std,
    )
    return _identity_calibrator()
