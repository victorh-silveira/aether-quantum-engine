"""Aplicacao do stacking tabular sobre trade_score com inversao micro no resolver."""

from __future__ import annotations

from typing import Any

from src.application.services.meta_classifier_cross_symbol import attach_cross_symbol_features_to_decisions
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.meta_classifier_client import (
    build_meta_predict_request,
    fallback_payoff_score,
    get_meta_classifier_client,
    meta_classifier_enabled,
    predict_meta_via_config_sync,
)


def apply_meta_payoff_to_metrics(
    metrics: dict[str, Any],
    *,
    direction: TradeDirection,
    tcn_probability: float,
    payoff_score: float,
    meta_applied: bool,
) -> float:
    """Atualiza trade_score e conviccao com score refinado do meta-classificador."""
    _ = tcn_probability
    score = max(0.0, min(1.0, float(payoff_score)))
    metrics["meta_calibrated_payoff_score"] = score
    metrics["meta_classifier_applied"] = bool(meta_applied)
    metrics["trade_score"] = score
    metrics["conviction"] = score
    if direction == TradeDirection.CALL:
        metrics["direction_call_score"] = score
        metrics["direction_put_score"] = max(0.0, 1.0 - score)
    else:
        metrics["direction_put_score"] = score
        metrics["direction_call_score"] = max(0.0, 1.0 - score)
    metrics["direction_margin"] = abs(metrics["direction_call_score"] - metrics["direction_put_score"])
    return score


async def prefetch_meta_payoff_for_decisions(decisions: dict[str, dict], config: dict[str, Any]) -> None:
    """Enriquece decisoes DL com stacking tabular em paralelo antes da coleta."""
    if not meta_classifier_enabled(config):
        return
    attach_cross_symbol_features_to_decisions(decisions)
    client = await get_meta_classifier_client(config)
    batch: list[tuple] = []
    refs: list[tuple[dict, TradeDirection, float]] = []
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
        fallback = fallback_payoff_score(metrics, str(direction), tcn_prob)
        request = build_meta_predict_request(
            symbol=str(symbol),
            metrics=metrics,
            tcn_probability=tcn_prob,
            direction=str(direction),
        )
        batch.append((request, fallback))
        refs.append((metrics, direction, tcn_prob))
    if not batch:
        return
    responses = await client.predict_meta_batch(batch)
    for (metrics, direction, tcn_prob), response in zip(refs, responses, strict=True):
        apply_meta_payoff_to_metrics(
            metrics,
            direction=direction,
            tcn_probability=tcn_prob,
            payoff_score=response["calibrated_payoff_score"],
            meta_applied=response["meta_applied"],
        )


def resolve_meta_payoff_score(
    *,
    symbol: str | None,
    metrics: dict[str, Any],
    direction: TradeDirection,
    tcn_probability: float,
    base_score: float,
    config: dict[str, Any] | None,
) -> tuple[float, bool]:
    """Resolve score meta-classificador com prefetch, chamada sync ou fallback TCN."""
    prefetched = metrics.get("meta_calibrated_payoff_score")
    if prefetched is not None:
        applied = bool(metrics.get("meta_classifier_applied", True))
        return max(0.0, min(1.0, float(prefetched))), applied
    if not config or not meta_classifier_enabled(config):
        return float(base_score), False
    fallback = fallback_payoff_score(metrics, direction.name, tcn_probability)
    request = build_meta_predict_request(
        symbol=str(symbol or metrics.get("symbol") or ""),
        metrics=metrics,
        tcn_probability=tcn_probability,
        direction=direction.name,
    )
    try:
        response = predict_meta_via_config_sync(config, request, fallback_score=fallback)
    except Exception:
        return float(fallback), False
    return (
        max(0.0, min(1.0, float(response["calibrated_payoff_score"]))),
        bool(response["meta_applied"]),
    )
