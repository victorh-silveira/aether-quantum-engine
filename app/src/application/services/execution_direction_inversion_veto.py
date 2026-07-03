"""Veto de inversao tatica quando a TCN tem alta conviccao direcional."""

from __future__ import annotations

from dataclasses import replace

from src.application.services.execution_universal_regime_types import RegimeEvaluation
from src.domain.models.trade import TradeDirection


def dl_side_probability(metrics: dict, dl_dir: TradeDirection) -> float:
    """Probabilidade calibrada alinhada ao lado previsto pela TCN."""
    raw = metrics.get("calibrated_prob")
    if raw is None:
        raw = metrics.get("raw_prob")
    if raw is None:
        score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        prob_call = score if dl_dir == TradeDirection.CALL else 1.0 - score
    else:
        prob_call = float(raw)
    return prob_call if dl_dir == TradeDirection.CALL else 1.0 - prob_call


def veto_inversion_on_dl_conviction(
    evaluation: RegimeEvaluation,
    metrics: dict,
    dl_dir: TradeDirection,
    *,
    veto_score: float,
) -> RegimeEvaluation:
    """Proibe inversao tatica quando P(lado_DL) >= veto_score; segue a TCN estrita."""
    if not evaluation.direction_inverted:
        return evaluation
    if dl_side_probability(metrics, dl_dir) + 1e-9 < float(veto_score):
        return evaluation
    metrics["dl_inversion_veto"] = True
    metrics["dl_side_probability"] = dl_side_probability(metrics, dl_dir)
    metrics["direction_inverted"] = False
    return replace(evaluation, direction_inverted=False, trap_boost_score=None)
