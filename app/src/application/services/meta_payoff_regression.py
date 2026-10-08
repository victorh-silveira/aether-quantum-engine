"""Decisao de stacking com edge continuo do meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import sync_direction_margin
from src.application.services.execution_runtime_config import resolve_meta_payoff_veto_config
from src.application.services.regime_micro_freeze import (
    log_d_squeeze_audit,
    micro_volatility_squeeze_active,
    severe_bb_compression,
)
from src.domain.models.trade import TradeDirection


NEUTRAL_AXIS = 0.5
CALIBRATION_NEUTRAL_DRIFT = "calibration_neutral_drift"
_META_SAT_EDGE = 0.85
_META_SAT_EPS = 1e-6


def calibration_neutral_axis_drift(raw_prob: float | None, calibrated_prob: float | None) -> bool:
    """True quando a calibracao inverte o lado macro em relacao ao eixo neutro 0.50."""
    if raw_prob is None or calibrated_prob is None:
        return False
    raw = float(raw_prob)
    calibrated = float(calibrated_prob)
    raw_call = raw > NEUTRAL_AXIS
    raw_put = raw < NEUTRAL_AXIS
    calibrated_call = calibrated > NEUTRAL_AXIS
    calibrated_put = calibrated < NEUTRAL_AXIS
    if raw_call and calibrated_put:
        return True
    return bool(raw_put and calibrated_call)


def _apply_direction_scores(metrics: dict[str, Any], *, direction: TradeDirection, score: float) -> None:
    """Propaga trade_score lateralizado para metricas de direcao CALL/PUT."""
    clamped = max(0.0, min(1.0, float(score)))
    metrics["trade_score"] = clamped
    metrics["conviction"] = clamped
    if direction == TradeDirection.CALL:
        metrics["direction_call_score"] = clamped
        metrics["direction_put_score"] = max(0.0, 1.0 - clamped)
    else:
        metrics["direction_put_score"] = clamped
        metrics["direction_call_score"] = max(0.0, 1.0 - clamped)
    sync_direction_margin(metrics, direction=direction.name)


def _clear_meta_soft_kelly(metrics: dict[str, Any]) -> None:
    """Limpa flags de soft Kelly do META neste ciclo."""
    metrics.pop("meta_soft_kelly", None)
    metrics.pop("meta_soft_strong", None)


def _meta_edge_saturated(predicted_edge: float) -> bool:
    """True quando edge META bate o clip de payout 0.85 (sat=1 nos logs)."""
    return abs(float(predicted_edge) - _META_SAT_EDGE) < _META_SAT_EPS


def _cal_side_edge_soft_for_sat(metrics: dict[str, Any], *, max_edge: float) -> bool:
    """True se cal_side_edge existe e e <= max_edge (sat META vs Cal mole/BE)."""
    raw = metrics.get("cal_side_edge")
    if raw is None:
        return False
    try:
        return float(raw) <= float(max_edge) + 1e-12
    except (TypeError, ValueError):
        return False


def apply_meta_regression_edge(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    predicted_edge: float,
    *,
    meta_applied: bool,
    base_score: float,
    symbol: str | None = None,
) -> tuple[TradeDirection, float]:
    """Aplica telemetria do meta-regressor sem soft Kelly (direcao = TCN)."""
    metrics["meta_classifier_applied"] = bool(meta_applied)
    if meta_applied:
        metrics["predicted_payoff_edge"] = float(predicted_edge)
    else:
        metrics.pop("predicted_payoff_edge", None)
    squeeze_active = micro_volatility_squeeze_active(metrics)
    metrics["meta_squeeze_active"] = bool(squeeze_active)
    _clear_meta_soft_kelly(metrics)
    if not meta_applied:
        metrics.pop("meta_negative_edge", None)
        metrics.pop("meta_sat_soft", None)
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        return dl_dir, float(base_score)
    if float(predicted_edge) <= 0.0:
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        metrics["meta_negative_edge"] = True
        metrics["meta_sat_soft"] = False
        return dl_dir, float(base_score)
    cfg = resolve_meta_payoff_veto_config()
    sat_cal_max = float(cfg["soft_veto_sat_cal_edge_max"])
    if _meta_edge_saturated(predicted_edge) and _cal_side_edge_soft_for_sat(metrics, max_edge=sat_cal_max):
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        metrics.pop("meta_negative_edge", None)
        metrics["meta_sat_soft"] = True
        return dl_dir, float(base_score)
    metrics.pop("meta_negative_edge", None)
    metrics.pop("meta_sat_soft", None)
    squeeze_danger = severe_bb_compression(metrics)
    if squeeze_danger:
        squeeze_score = float(cfg["squeeze_trade_score"])
        metrics["meta_squeeze_downgrade"] = True
        _apply_direction_scores(metrics, direction=dl_dir, score=squeeze_score)
        log_d_squeeze_audit(symbol, metrics)
        return dl_dir, squeeze_score
    _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
    return dl_dir, float(base_score)
