"""Aplicacao do stacking tabular com edge continuo do meta-regressor."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.meta_classifier_client import (
    build_meta_predict_request,
    fallback_payoff_score,
    meta_classifier_enabled,
)
from src.infrastructure.inference.meta_classifier_pool import get_meta_classifier_client


def apply_meta_regression_edge_to_metrics(
    metrics: dict[str, Any],
    *,
    direction: TradeDirection,
    tcn_probability: float,
    predicted_edge: float,
    meta_applied: bool,
    base_score: float,
) -> float:
    """Anexa edge continuo e preserva trade_score organico da TCN no prefetch."""
    _ = (direction, tcn_probability)
    metrics["predicted_payoff_edge"] = float(predicted_edge)
    metrics["meta_classifier_applied"] = bool(meta_applied)
    score = float(base_score)
    metrics["trade_score"] = max(0.0, min(1.0, score))
    metrics["conviction"] = metrics["trade_score"]
    if direction == TradeDirection.CALL:
        metrics["direction_call_score"] = metrics["trade_score"]
        metrics["direction_put_score"] = max(0.0, 1.0 - metrics["trade_score"])
    else:
        metrics["direction_put_score"] = metrics["trade_score"]
        metrics["direction_call_score"] = max(0.0, 1.0 - metrics["trade_score"])
    metrics["direction_margin"] = abs(metrics["direction_call_score"] - metrics["direction_put_score"])
    return metrics["trade_score"]


async def prefetch_meta_payoff_for_decisions(decisions: dict[str, dict], config: dict[str, Any]) -> None:
    """Enriquece decisoes DL com edge continuo do meta-regressor em paralelo."""
    if not meta_classifier_enabled(config):
        return
    client = await get_meta_classifier_client(config)
    batch: list[tuple] = []
    refs: list[tuple[dict, TradeDirection, float, float]] = []
    for symbol, entry in decisions.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        direction = entry.get("direction")
        if direction is None:
            continue
        prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
        if prob is None:
            continue
        tcn_prob = float(prob)
        base_score = fallback_payoff_score(metrics, direction.name, tcn_prob)
        request = build_meta_predict_request(
            symbol=str(symbol),
            metrics=metrics,
            tcn_probability=tcn_prob,
            direction=direction.name,
        )
        batch.append((request, base_score))
        refs.append((metrics, direction, tcn_prob, base_score))
    if not batch:
        return
    responses = await client.predict_meta_batch(batch)
    for (metrics, direction, tcn_prob, base_score), response in zip(refs, responses, strict=True):
        apply_meta_regression_edge_to_metrics(
            metrics,
            direction=direction,
            tcn_probability=tcn_prob,
            predicted_edge=response["predicted_payoff_edge"],
            meta_applied=response["meta_applied"],
            base_score=base_score,
        )


def resolve_meta_payoff_edge(
    *,
    symbol: str | None,
    metrics: dict[str, Any],
    direction: TradeDirection,
    tcn_probability: float,
    _base_score: float,
    config: dict[str, Any] | None,
) -> tuple[float, bool]:
    """Resolve edge continuo apenas a partir do prefetch do ciclo M1."""
    _ = (symbol, direction, tcn_probability, _base_score, config)
    prefetched = metrics.get("predicted_payoff_edge")
    if prefetched is not None:
        applied = bool(metrics.get("meta_classifier_applied", True))
        return float(prefetched), applied
    return 0.0, False
