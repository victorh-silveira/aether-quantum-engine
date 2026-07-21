"""Tolerancia de calibracao e override TCN macro sem zona de abstencao."""

from __future__ import annotations

import json

from aether_paths import repo_path
from src.domain.models.trade import TradeDirection


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
    if raw > float(_tol()["tcn_macro_call_override"]):
        resolved = direction if direction is not None else TradeDirection.CALL
        return raw, resolved, "tcn_macro_override"
    if raw < float(_tol()["tcn_macro_put_override"]):
        resolved = direction if direction is not None else TradeDirection.PUT
        return raw, resolved, "tcn_macro_override"
    if direction is not None:
        return cal, direction, "calibrated"
    return cal, infer_direction_from_prob(cal, None, pivot=pivot), "calibrated"
