"""Regras de gating e calculo de edge EV para inferencia do Deep Learning."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


MARKET_PAYOUT_SSOT = 0.85


def direction_from_raw_prob(raw_prob: float, call_threshold: float = 0.55, put_threshold: float = 0.45) -> Any:
    """Decodifica direcao CALL/PUT a partir da probabilidade raw e thresholds."""
    if raw_prob >= call_threshold:
        return TradeDirection.CALL
    if raw_prob <= put_threshold:
        return TradeDirection.PUT
    return None


def _adjust_payout_for_horizon(payout: float, horizon_bars: int = 4) -> float:
    """Ajusta o payout esperado baseado no horizonte de predicao multi-candle."""
    if horizon_bars <= 1:
        return payout
    decay = 1.0 - (0.015 * float(horizon_bars - 1))
    return max(0.80, payout * decay)


def resolve_edge(prob: float, payout: float = MARKET_PAYOUT_SSOT, horizon_bars: int = 1) -> float:
    """Edge EV assinado do lado dominante: p_win*(1+b)-1 (pode ser negativo)."""
    if prob is None:
        return 0.0
    p = float(prob)
    p_win = max(p, 1.0 - p)
    adj_payout = _adjust_payout_for_horizon(payout, horizon_bars)
    return float((p_win * (1.0 + adj_payout)) - 1.0)


def resolve_calibrated_edge(
    calibrated_prob: float | None,
    raw_prob: float | None = 0.5,
    payout: float = MARKET_PAYOUT_SSOT,
    horizon_bars: int = 1,
) -> float:
    """Edge EV assinado a partir da probabilidade calibrada (lado dominante)."""
    if calibrated_prob is None:
        return resolve_edge(raw_prob, payout, horizon_bars=horizon_bars)

    p = float(calibrated_prob)
    p_win = max(p, 1.0 - p)
    adj_payout = _adjust_payout_for_horizon(payout, horizon_bars)
    return float((p_win * (1.0 + adj_payout)) - 1.0)


def resolve_side_edge(
    calibrated_prob: float,
    *,
    direction: Any,
    payout: float = MARKET_PAYOUT_SSOT,
    horizon_bars: int = 1,
) -> float:
    """Edge EV do lado TCN (CALL usa p_cal; PUT usa 1-p_cal)."""
    p = float(calibrated_prob)
    name = getattr(direction, "name", None) or str(direction or "").upper()
    if str(name).upper() == "PUT":
        p = 1.0 - p
    adj_payout = _adjust_payout_for_horizon(payout, horizon_bars)
    return float((p * (1.0 + adj_payout)) - 1.0)


def resolve_confidence_thresholds(params: dict) -> tuple[float, float]:
    """Retorna thresholds CALL e PUT para o movimento de mercado."""
    if not isinstance(params, dict):
        return (0.55, 0.45)
    return (
        float(params.get("confidence_call_threshold", 0.55)),
        float(params.get("confidence_put_threshold", 0.45)),
    )
