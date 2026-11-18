"""Metodos auxiliares de calibracao DL: isotonica, variancia, tolerancia e nitidez."""

from __future__ import annotations

import json
import logging

from src.application.services.deep_learning.dl_calibration import (
    _METHOD_IDENTITY,
    _METHOD_TEMPERATURE,
    CalibratorState,
    apply_calibrator,
    apply_calibrator_stable,
    apply_isotonic,
    temperature_bounds,
)
from src.application.services.deep_learning.dl_sharpness import mean_sharpness
from src.domain.math.logit_sharpen import (
    apply_dynamic_temperature_sharpening,
    logit,
    resolve_sharpening_tau,
    sharpen_logit_temperature,
    sigmoid,
)
from src.domain.models.trade import TradeDirection
from src.shared.paths import repo_path


__all__ = (
    "apply_calibration_neutral_tolerance",
    "apply_dynamic_temperature_sharpening",
    "apply_isotonic",
    "fit_isotonic",
    "infer_direction_from_prob",
    "logit",
    "maybe_identity_on_variance_collapse",
    "maybe_temperature_sharpen_for_export",
    "resolve_sharpening_tau",
    "sharpen_logit_temperature",
    "sigmoid",
)

logger = logging.getLogger("AETH")

_CAL_VARIANCE_COLLAPSE_STD = 1e-3
_CAL_VARIANCE_MIN_STD_RATIO = 0.5


def fit_isotonic(probs: list[float], labels: list[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Ajusta curva monotona via PAV sobre pares (prob, label) ordenados."""
    if not probs:
        return (), ()
    order = sorted(range(len(probs)), key=lambda idx: float(probs[idx]))
    xs = [float(probs[idx]) for idx in order]
    ys = [float(labels[idx]) for idx in order]
    weights = [1.0] * len(xs)
    idx = 0
    while idx < len(ys) - 1:
        if ys[idx] <= ys[idx + 1] + 1e-12:
            idx += 1
            continue
        merged_w = weights[idx] + weights[idx + 1]
        merged_y = (ys[idx] * weights[idx] + ys[idx + 1] * weights[idx + 1]) / merged_w
        xs[idx] = max(xs[idx], xs[idx + 1])
        ys[idx] = merged_y
        weights[idx] = merged_w
        del xs[idx + 1]
        del ys[idx + 1]
        del weights[idx + 1]
        if idx > 0:
            idx -= 1
    return tuple(xs), tuple(ys)


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


def _tol() -> dict[str, float]:
    """Le tolerancia e overrides TCN macro de settings."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    raw = (full.get("deep_learning") or {}).get("calibration") or {}
    for key in ("neutral_calibration_half_width", "tcn_macro_call_override", "tcn_macro_put_override"):
        if key not in raw:
            raise ValueError(f"deep_learning.calibration.{key} obrigatorio")
    return {
        "neutral_calibration_half_width": float(raw["neutral_calibration_half_width"]),
        "tcn_macro_call_override": float(raw["tcn_macro_call_override"]),
        "tcn_macro_put_override": float(raw["tcn_macro_put_override"]),
    }


def _horizon_gap_bars() -> int:
    """Le o gap de barras entre o horizonte do label e a abertura do contrato."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") or {}
    return max(0, int(dl.get("horizon_gap_bars", 1)))


def infer_direction_from_prob(
    calibrated_prob: float,
    direction: TradeDirection | None,
    pivot: float = 0.5,
) -> TradeDirection:
    """Infere CALL ou PUT a partir da probabilidade calibrada."""
    if direction is not None:
        return direction
    return TradeDirection.CALL if float(calibrated_prob) + 1e-12 >= float(pivot) else TradeDirection.PUT


def apply_calibration_neutral_tolerance(
    calibrated_prob: float,
    raw_prob: float,
    direction: TradeDirection | None,
    *,
    pivot: float = 0.5,
    neutral_lo: float | None = None,
    neutral_hi: float | None = None,
) -> tuple[float, TradeDirection, str]:
    """Resolve CALL/PUT pela probabilidade final; raw_extreme e apenas telemetria."""
    del direction, neutral_lo, neutral_hi
    raw = float(raw_prob)
    cal = float(calibrated_prob)
    tol = _tol()
    extreme = raw > float(tol["tcn_macro_call_override"]) or raw < float(tol["tcn_macro_put_override"])
    return cal, infer_direction_from_prob(cal, None, pivot=pivot), "raw_extreme" if extreme else "calibrated"


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
