"""Quatro vetos extremos e reavaliacao direcional sem fabricar probabilidades."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.execution_price_action import _resolve_candle_ohlc
from src.application.services.execution_signal_skips import _mark_skip
from src.application.services.market_audit_candle import _all_closed_micro_candles, candle_binary_side
from src.domain.models.trade import TradeDirection


def resolve_four_vetoes_enabled(config: dict | None) -> bool:
    """Ativacao explicita; strings truthy nao ativam a politica."""
    return (config or {}).get("four_market_vetoes") is True


def _number(value: Any) -> float | None:
    """Le numero finito sem coercao de valores ausentes para zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _previous_closed_side(metrics: dict, orch: Any, symbol: str | None) -> str | None:
    """Usa duas velas distintas; SCALE prev pode apontar para a ultima fechada."""
    if orch is None:
        return metrics.get("previous_closed_candle_direction")
    candles = _all_closed_micro_candles(getattr(orch, "stream", None), str(symbol))
    if len(candles) < 2 or candles[-2].epoch >= candles[-1].epoch:
        return None
    return candle_binary_side(candles[-2])


def market_veto_reason(
    direction: TradeDirection, metrics: dict, *, orch: Any = None, symbol: str | None = None
) -> str | None:
    """Exige confluencia extrema, nao apenas tendencia ou edge fraco."""
    candle = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if candle is None:
        return None
    o, h, low, c = candle
    if not all(math.isfinite(v) and v > 0 for v in candle) or not low <= min(o, c) <= max(o, c) <= h:
        return None
    span = h - low
    if span <= 0:
        return None
    indicators = metrics.get("indicators")
    indicators = indicators if isinstance(indicators, dict) else metrics
    rsi = _number(indicators.get("rsi"))
    bb = _number(indicators.get("bb_pct_b"))
    di = _number(indicators.get("di_diff"))
    call = direction == TradeDirection.CALL
    wick = (h - max(o, c)) / span if call else (min(o, c) - low) / span
    extreme = (
        rsi is not None and bb is not None and ((rsi >= 0.75 and bb >= 1.0) if call else (rsi <= 0.25 and bb <= 0.0))
    )
    if extreme and wick >= 0.45:
        return "call_top_rejection" if call else "put_bottom_rejection"
    opposite = "PUT" if call else "CALL"
    momentum = di is not None and (di <= -0.15 if call else di >= 0.15)
    body_opposite = c < o if call else c > o
    if (
        momentum
        and body_opposite
        and abs(c - o) / span >= 0.8
        and metrics.get("trend_direction") == opposite
        and _previous_closed_side(metrics, orch, symbol) == opposite
    ):
        return "call_down_continuation" if call else "put_up_continuation"
    return None


def apply_four_market_vetoes(direction: TradeDirection, metrics: dict, **context: Any) -> bool:
    """Aplica somente um dos quatro motivos de mercado; guardas tecnicas sao externas."""
    reason = market_veto_reason(direction, metrics, **context)
    if reason is None:
        return False
    _mark_skip(metrics, reason)
    return True


def reevaluate_market_direction(
    direction: TradeDirection,
    metrics: dict,
    config: dict | None,
    *,
    payout: float,
    orch: Any = None,
    symbol: str | None = None,
) -> TradeDirection:
    """Reavalia lado com probabilidade calibrada existente e EV positivo do candidato."""
    metrics["market_trigger_applied"] = False
    if not resolve_four_vetoes_enabled(config) or (config or {}).get("market_direction_trigger") is not True:
        return direction
    reason = market_veto_reason(direction, metrics, orch=orch, symbol=symbol)
    if reason is None:
        metrics["market_trigger_status"] = "no_extreme_setup"
        return direction
    candidate = TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL
    metrics["market_trigger_candidate"] = candidate.name
    metrics["market_trigger_setup"] = reason
    prob = _number(metrics.get("calibrated_prob"))
    rate = _number(payout)
    floor = _number((config or {}).get("min_edge_execute", 0.01))
    if prob is None or not 0 <= prob <= 1 or rate is None or rate <= 0 or floor is None or floor < 0:
        metrics["market_trigger_status"] = "invalid_probability_or_payout"
        return direction
    side_prob = prob if candidate == TradeDirection.CALL else 1 - prob
    edge = side_prob * (1 + rate) - 1
    metrics["market_trigger_candidate_edge"] = edge
    if edge <= floor:
        metrics["market_trigger_status"] = "candidate_without_edge"
        return direction
    if market_veto_reason(candidate, metrics, orch=orch, symbol=symbol) is not None:
        metrics["market_trigger_status"] = "candidate_vetoed"
        return direction
    metrics.update(
        market_trigger_applied=True,
        market_trigger_status="accepted_model_supported",
        direction_origin="MARKET_TRIGGER_TCN",
        cal_side_edge=edge,
        exec_direction=candidate.name,
        resolved_direction=candidate.name,
        conviction=side_prob,
        trade_score=side_prob,
        loss_clf_flip=False,
        anti_trend_lock_flip=False,
    )
    return candidate
