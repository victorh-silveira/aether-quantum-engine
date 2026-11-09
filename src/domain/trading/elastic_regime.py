"""Modelo de dominio para classificacao elastica de regime e resolucao direcional."""

from __future__ import annotations

from enum import StrEnum


class MarketRegime(StrEnum):
    """Regimes tecnicos de mercado avaliados em tempo real."""

    STRONG_TREND = "STRONG_TREND"
    MODERATE_TREND = "MODERATE_TREND"
    RANGING = "RANGING"
    EXHAUSTION = "EXHAUSTION"


def classify_market_regime(
    adx: float,
    spread: float,
    rsi: float = 50.0,
    recuo: float = 0.0,
) -> MarketRegime:
    """Classifica o regime de mercado a partir de indicadores tecnicos."""
    clean_rsi = float(rsi) if isinstance(rsi, (int, float)) else 50.0
    clean_recuo = float(recuo) if isinstance(recuo, (int, float)) else 0.0
    clean_adx = float(adx) if isinstance(adx, (int, float)) else 0.0
    clean_spread = float(spread) if isinstance(spread, (int, float)) else 0.0

    if clean_adx >= 22.0 or abs(clean_spread) > 0.01:
        return MarketRegime.STRONG_TREND
    if (clean_rsi > 75.0 or clean_rsi < 25.0) or clean_recuo > 2.0:
        return MarketRegime.EXHAUSTION
    if 18.0 <= clean_adx < 22.0:
        return MarketRegime.MODERATE_TREND
    if 0.0 < clean_adx < 18.0:
        return MarketRegime.RANGING
    return MarketRegime.MODERATE_TREND


def evaluate_elastic_direction(
    original_side: str,
    trend: str,
    regime: MarketRegime,
    *,
    has_loss_reversal: bool = False,
) -> tuple[str, str, bool]:
    """Determina a direcao final com base no regime sem listas estaticas."""
    side = str(original_side).upper().strip()
    if side not in ("CALL", "PUT"):
        return side, "INVALID_SIDE", False

    opp_side = "PUT" if side == "CALL" else "CALL"
    trend_clean = str(trend).upper().strip()
    is_aligned = (trend_clean == "ALTA" and side == "CALL") or (trend_clean == "BAIXA" and side == "PUT")

    if is_aligned:
        return side, "TREND_ALIGNED_DIRECT", False

    if regime == MarketRegime.STRONG_TREND:
        if has_loss_reversal:
            return opp_side, "REVERSAL_ON_LOSS", True
        return side, "BASE_SIGNAL_DIRECT", False

    if regime == MarketRegime.EXHAUSTION:
        return opp_side, "ELASTIC_EXHAUSTION_INVERTED", True

    if regime == MarketRegime.RANGING:
        if has_loss_reversal:
            return opp_side, "REVERSAL_ON_LOSS", True
        return opp_side, "LATERAL_FADE_INVERTED", True

    if has_loss_reversal:
        return opp_side, "REVERSAL_ON_LOSS", True
    return side, "BASE_SIGNAL_DIRECT", False
