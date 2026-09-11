"""Decisao de stacking com edge continuo do meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_gate_verdict import stamp_soft_size
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
    metrics.pop("meta_negative_edge", None)
    metrics.pop("meta_soft_kelly", None)
    metrics.pop("meta_sat_soft", None)
    metrics.pop("meta_soft_strong", None)


def _resolve_soft_factor(predicted_edge: float, cfg: dict[str, Any]) -> float:
    """Escolhe fator soft normal ou forte conforme edge META."""
    mild = max(0.05, min(1.0, float(cfg["soft_veto_score_factor"])))
    strong_edge = float(cfg["soft_veto_strong_edge"])
    if float(predicted_edge) <= strong_edge + 1e-12:
        return max(0.05, min(mild, float(cfg["soft_veto_strong_factor"])))
    return mild


def _apply_meta_soft_kelly(
    metrics: dict[str, Any],
    *,
    reason: str,
    predicted_edge: float,
) -> None:
    """Comprime kelly_fraction_scale via soft_veto_* (sem SKIP)."""
    cfg = resolve_meta_payoff_veto_config()
    factor = _resolve_soft_factor(float(predicted_edge), cfg)
    current = float(metrics.get("kelly_fraction_scale", 1.0) or 1.0)
    metrics["kelly_fraction_scale"] = max(0.05, current * factor)
    metrics["meta_soft_kelly"] = True
    metrics["meta_soft_strong"] = bool(float(predicted_edge) <= float(cfg["soft_veto_strong_edge"]) + 1e-12)
    stamp_soft_size(metrics, reason)


def _meta_edge_saturated(predicted_edge: float) -> bool:
    """True quando edge META bate o clip de payout 0.85 (sat=1 nos logs)."""
    return abs(float(predicted_edge) - _META_SAT_EDGE) < _META_SAT_EPS


def _cal_side_edge_nonpositive(metrics: dict[str, Any]) -> bool:
    """True se cal_side_edge existe e e <= 0 (Edge CLUSTER negativo/zero)."""
    raw = metrics.get("cal_side_edge")
    if raw is None:
        return False
    try:
        return float(raw) <= 0.0
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
    """Aplica edge continuo do meta-regressor com soft Kelly se edge <= 0 ou sat vs Cal."""
    metrics["meta_classifier_applied"] = bool(meta_applied)
    if meta_applied:
        metrics["predicted_payoff_edge"] = float(predicted_edge)
    else:
        metrics.pop("predicted_payoff_edge", None)
    squeeze_active = micro_volatility_squeeze_active(metrics)
    metrics["meta_squeeze_active"] = bool(squeeze_active)
    if not meta_applied:
        _clear_meta_soft_kelly(metrics)
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        return dl_dir, float(base_score)
    if float(predicted_edge) <= 0.0:
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        metrics["meta_negative_edge"] = True
        metrics["meta_sat_soft"] = False
        _apply_meta_soft_kelly(metrics, reason="meta_negative_edge", predicted_edge=float(predicted_edge))
        return dl_dir, float(base_score)
    if _meta_edge_saturated(predicted_edge) and _cal_side_edge_nonpositive(metrics):
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        metrics.pop("meta_negative_edge", None)
        metrics["meta_sat_soft"] = True
        _apply_meta_soft_kelly(metrics, reason="meta_sat_vs_neg_cal", predicted_edge=float(predicted_edge))
        return dl_dir, float(base_score)
    _clear_meta_soft_kelly(metrics)
    squeeze_danger = severe_bb_compression(metrics)
    if squeeze_danger:
        squeeze_score = float(resolve_meta_payoff_veto_config()["squeeze_trade_score"])
        metrics["meta_squeeze_downgrade"] = True
        _apply_direction_scores(metrics, direction=dl_dir, score=squeeze_score)
        log_d_squeeze_audit(symbol, metrics)
        return dl_dir, squeeze_score
    _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
    return dl_dir, float(base_score)
