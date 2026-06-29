"""Composicao de scores CALL/PUT no resolver direcional."""

from __future__ import annotations

import contextlib

from src.application.services.execution_direction_hints import indicator_regime_side
from src.domain.models.trade import TradeDirection


def accumulate_direction_scores(
    entry: dict,
    metrics: dict,
    weights: dict,
    *,
    recovery_active: bool,
    bias_fns: tuple,
    dl_scores_fn,
    val_bias_fn,
    low_val_fn,
) -> tuple[float, float, list[str]]:
    """Soma contribuicoes laterais e retorna scores e hints."""
    call_score = 0.0
    put_score = 0.0
    hints: list[str] = []
    dl_call, dl_put = dl_scores_fn(entry, weights)
    call_score += dl_call
    put_score += dl_put
    val_call, val_put = val_bias_fn(metrics, weights)
    call_score += val_call - 0.5
    put_score += val_put - 0.5
    for bias_fn in bias_fns:
        c_bias, p_bias, hint = bias_fn(metrics, weights)
        call_score += c_bias - 0.5
        put_score += p_bias - 0.5
        if hint and hint not in hints:
            hints.append(hint)
    indicator_regime_side(metrics)
    c_low, p_low, low_hint = low_val_fn(entry, metrics, weights)
    call_score += c_low - 0.5
    put_score += p_low - 0.5
    if low_hint and low_hint not in hints:
        hints.append(low_hint)
    if recovery_active:
        trend_str = metrics.get("trend_direction")
        if trend_str:
            with contextlib.suppress(KeyError, ValueError):
                trend_dir = TradeDirection[str(trend_str).upper()]
                w = float(weights["trend_weight"]) * 0.5
                if trend_dir == TradeDirection.CALL:
                    call_score += w
                else:
                    put_score += w
    return call_score, put_score, hints


def finalize_direction_metrics(
    metrics: dict,
    *,
    call_score: float,
    put_score: float,
    hints: list[str],
    dl_dir: TradeDirection,
    exec_dir: TradeDirection,
    clamp01,
) -> None:
    """Grava scores e conviccao resolvida em metrics."""
    chosen = max(call_score, put_score)
    metrics["direction_call_score"] = call_score
    metrics["direction_put_score"] = put_score
    metrics["direction_margin"] = abs(call_score - put_score)
    metrics["direction_hints"] = hints
    metrics["direction_hint"] = hints[0] if hints else None
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    metrics["direction_inverted"] = dl_dir != exec_dir
    side_strength = clamp01(chosen)
    if metrics.get("trade_score") is None:
        metrics["trade_score"] = side_strength
    metrics["resolved_conviction"] = side_strength
