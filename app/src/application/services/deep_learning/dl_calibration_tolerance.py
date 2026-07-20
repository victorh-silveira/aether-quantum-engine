"""Tolerancia de calibracao e override TCN macro sem zona de abstencao."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


NEUTRAL_CALIBRATION_HALF_WIDTH = 0.0
TCN_MACRO_CALL_OVERRIDE = 0.65
TCN_MACRO_PUT_OVERRIDE = 0.35


def infer_direction_from_prob(
    calibrated_prob: float,
    direction: TradeDirection | None,
    pivot: float = 0.5,
) -> TradeDirection:
    """Infere CALL ou PUT a partir da probabilidade calibrada quando direction e None."""
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
) -> tuple[float, TradeDirection | None, str]:
    """Aplica override TCN macro e sempre resolve CALL/PUT (sem abstencao neutra)."""
    _ = (neutral_lo, neutral_hi)
    raw = float(raw_prob)
    cal = float(calibrated_prob)
    if raw > TCN_MACRO_CALL_OVERRIDE:
        resolved = direction if direction is not None else TradeDirection.CALL
        return raw, resolved, "tcn_macro_override"
    if raw < TCN_MACRO_PUT_OVERRIDE:
        resolved = direction if direction is not None else TradeDirection.PUT
        return raw, resolved, "tcn_macro_override"
    if direction is not None:
        return cal, direction, "calibrated"
    return cal, infer_direction_from_prob(cal, None, pivot=pivot), "calibrated"
