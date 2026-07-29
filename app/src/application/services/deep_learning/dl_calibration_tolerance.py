"""Tolerancia de calibracao para previsao do movimento de mercado multi-candle."""

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


def _in_neutral_zone(cal: float, neutral_lo: float | None, neutral_hi: float | None) -> bool:
    """Verifica se a probabilidade calibrada cai na zona neutra."""
    if neutral_lo is not None and neutral_hi is not None:
        lo = min(float(neutral_lo), float(neutral_hi))
        hi = max(float(neutral_lo), float(neutral_hi))
        return lo <= float(cal) <= hi
    return False


def apply_calibration_neutral_tolerance(
    calibrated_prob: float,
    raw_prob: float,
    direction: TradeDirection | None,
    *,
    pivot: float = 0.5,
    neutral_lo: float | None = None,
    neutral_hi: float | None = None,
) -> tuple[float, TradeDirection | None, str]:
    """Aplica override TCN macro e zona neutra para movimento multi-candle."""
    raw = float(raw_prob)
    cal = float(calibrated_prob)
    tol = _tol()
    half_width = float(tol["neutral_calibration_half_width"])
    effective_neutral_lo = neutral_lo if neutral_lo is not None else 0.5 - half_width
    effective_neutral_hi = neutral_hi if neutral_hi is not None else 0.5 + half_width
    if raw > float(tol["tcn_macro_call_override"]):
        resolved = direction if direction is not None else TradeDirection.CALL
        return raw, resolved, "tcn_macro_override"
    if raw < float(tol["tcn_macro_put_override"]):
        resolved = direction if direction is not None else TradeDirection.PUT
        return raw, resolved, "tcn_macro_override"
    if direction is not None:
        return cal, direction, "calibrated"
    if _in_neutral_zone(cal, effective_neutral_lo, effective_neutral_hi):
        return cal, None, "neutral_zone"
    return cal, infer_direction_from_prob(cal, None, pivot=pivot), "calibrated"
