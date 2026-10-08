"""Classificador e ensemble de regime de mercado condicionado por volatilidade e microestrutura."""

from __future__ import annotations

from typing import Any


REGIME_COMPRESSION_RISK = "COMPRESSION_RISK"
REGIME_EXPLOSION_MOMENTUM = "EXPLOSION_MOMENTUM"
REGIME_BALANCED = "BALANCED"

COMPRESSION_RV_CEILING = 0.40
EXPLOSION_RV_FLOOR = 1.60
HURST_MEAN_REVERTING_CEILING = 0.45
ADX_TREND_THRESHOLD = 0.25


def classify_market_regime(
    rv_ratio: float | None,
    *,
    adx: float | None = None,
    hurst: float | None = None,
    bb_width: float | None = None,
) -> dict[str, Any]:
    """Classifica o regime operacional em compressao/doji, explosao ou equilibrado."""
    ratio = float(rv_ratio) if rv_ratio is not None else 1.0
    val_adx = float(adx) if adx is not None else 0.20
    val_hurst = float(hurst) if hurst is not None else 0.50
    val_bbw = float(bb_width) if bb_width is not None else 0.05

    is_compression = ratio <= COMPRESSION_RV_CEILING or (val_hurst < HURST_MEAN_REVERTING_CEILING and val_bbw < 0.03)
    is_explosion = ratio >= EXPLOSION_RV_FLOOR and val_adx >= ADX_TREND_THRESHOLD

    if is_compression:
        regime = REGIME_COMPRESSION_RISK
        recommendation = "veto_breakout"
        doji_risk = True
        momentum_boost = False
    elif is_explosion:
        regime = REGIME_EXPLOSION_MOMENTUM
        recommendation = "prioritize_trend"
        doji_risk = False
        momentum_boost = True
    else:
        regime = REGIME_BALANCED
        recommendation = "standard"
        doji_risk = False
        momentum_boost = False

    return {
        "regime": regime,
        "recommendation": recommendation,
        "doji_risk": doji_risk,
        "momentum_boost": momentum_boost,
        "rv_ratio": ratio,
        "adx": val_adx,
        "hurst": val_hurst,
        "bb_width": val_bbw,
    }


def apply_regime_ensemble_gate(
    metrics: dict[str, Any] | None,
    rv_ratio: float | None,
    *,
    adx: float | None = None,
    hurst: float | None = None,
    bb_width: float | None = None,
    veto_on_compression: bool = True,
) -> tuple[bool, str]:
    """Avalia o regime e decora metricas do ciclo, vetando se houver risco critico de doji."""
    info = classify_market_regime(rv_ratio, adx=adx, hurst=hurst, bb_width=bb_width)
    if isinstance(metrics, dict):
        metrics["regime_ensemble"] = info["regime"]
        metrics["regime_ensemble_recommendation"] = info["recommendation"]
        metrics["regime_ensemble_doji_risk"] = info["doji_risk"]
        metrics["regime_ensemble_momentum_boost"] = info["momentum_boost"]

    if veto_on_compression and info["regime"] == REGIME_COMPRESSION_RISK:
        return True, "regime_compression_doji_risk"
    return False, "ok"
