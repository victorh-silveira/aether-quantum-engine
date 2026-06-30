"""Funcoes de bias lateral do resolver direcional."""

from __future__ import annotations

import contextlib

from src.domain.models.trade import TradeDirection


def _clamp01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def val_accuracy_bias(metrics: dict, weights: dict) -> tuple[float, float]:
    """Aplica bias de val_accuracy ao score lateral."""
    val = _clamp01(float(metrics.get("val_accuracy", 0.5)))
    w = float(weights["val_accuracy_weight"])
    bias = (val - 0.5) * w
    return 0.5 + bias, 0.5 - bias


def trend_bias(metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Ajusta score conforme alinhamento com tendencia de mercado."""
    trend_str = metrics.get("trend_direction")
    if not trend_str:
        return 0.5, 0.5, None
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    adx = float(indicators.get("adx", 0.5))
    if vol_ratio < 0.85 and adx < 0.25:
        return 0.5, 0.5, None
    w = float(weights["trend_weight"])
    with contextlib.suppress(KeyError, ValueError):
        trend_dir = TradeDirection[str(trend_str).upper()]
        if trend_dir == TradeDirection.CALL:
            return 0.5 + w, 0.5 - w, "trend_bias"
        return 0.5 - w, 0.5 + w, "trend_bias"
    return 0.5, 0.5, None


def exhaustion_bias(
    metrics: dict,
    weights: dict,
    *,
    cfg: dict | None = None,
) -> tuple[float, float, str | None]:
    """Empurra direcao oposta em extremos de RSI/Keltner configuraveis."""
    gate = (cfg or {}).get("exhaustion_gate") if isinstance(cfg, dict) else {}
    gate = gate if isinstance(gate, dict) else {}
    indicators = metrics.get("indicators") or {}
    rsi = float(indicators.get("rsi", 0.5))
    keltner = float(indicators.get("keltner", 0.5))
    rsi_os = float(gate.get("rsi_oversold", 0.28))
    rsi_ob = float(gate.get("rsi_overbought", 0.73))
    k_os = float(gate.get("keltner_oversold", -0.15))
    k_ob = float(gate.get("keltner_overbought", 1.15))
    w = float(weights["exhaustion_weight"])
    if rsi < rsi_os or keltner < k_os:
        return 0.5 + w, 0.5 - w, "exhaustion_flip"
    if rsi > rsi_ob or keltner > k_ob:
        return 0.5 - w, 0.5 + w, "exhaustion_flip"
    return 0.5, 0.5, None


def indicator_regime_bias(metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Aplica pesos de regime (hurst, adx, vol_ratio, cmo) ao score."""
    indicators = metrics.get("indicators") or {}
    hurst = float(indicators.get("hurst", 0.5))
    adx = float(indicators.get("adx", 0.5))
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    rsi = float(indicators.get("rsi", 0.5))
    w = float(weights["indicator_regime_weight"])
    if hurst < 0.48 and adx < 0.25 and vol_ratio < 1.0:
        if rsi < 0.45:
            return 0.5 + w, 0.5 - w, "mean_reversion"
        if rsi > 0.55:
            return 0.5 - w, 0.5 + w, "mean_reversion"
    cmo = float(indicators.get("cmo", 0.0))
    if cmo > 0.08:
        return 0.5 + w * 0.5, 0.5 - w * 0.5, "indicator_regime"
    if cmo < -0.08:
        return 0.5 - w * 0.5, 0.5 + w * 0.5, "indicator_regime"
    return 0.5, 0.5, None
