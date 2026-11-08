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
