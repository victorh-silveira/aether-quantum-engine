"""Decisao de stacking com edge continuo do meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.meta_direction_flip import log_d_squeeze_audit, micro_volatility_squeeze_active
from src.domain.models.trade import TradeDirection


META_SEVERE_NEGATIVE_EDGE = -0.15
META_SQUEEZE_TRADE_SCORE = 0.52


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
    metrics["direction_margin"] = abs(metrics["direction_call_score"] - metrics["direction_put_score"])


def apply_meta_regression_edge(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    predicted_edge: float,
    *,
    meta_applied: bool,
    base_score: float,
    symbol: str | None = None,
) -> tuple[TradeDirection, float]:
    """Aplica edge continuo do meta-regressor com downgrade D-SQUEEZE quando necessario."""
    metrics["predicted_payoff_edge"] = float(predicted_edge)
    metrics["meta_classifier_applied"] = bool(meta_applied)
    squeeze_active = micro_volatility_squeeze_active(metrics)
    metrics["meta_squeeze_active"] = bool(squeeze_active)
    if not meta_applied:
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        return dl_dir, float(base_score)
    if float(predicted_edge) > 0.0:
        _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
        return dl_dir, float(base_score)
    if float(predicted_edge) < META_SEVERE_NEGATIVE_EDGE and squeeze_active:
        metrics["meta_squeeze_downgrade"] = True
        _apply_direction_scores(metrics, direction=dl_dir, score=META_SQUEEZE_TRADE_SCORE)
        log_d_squeeze_audit(symbol, metrics)
        return dl_dir, META_SQUEEZE_TRADE_SCORE
    _apply_direction_scores(metrics, direction=dl_dir, score=base_score)
    return dl_dir, float(base_score)
