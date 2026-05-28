"""Pontuacao de alinhamento Z-Score StatArb com direcao do cluster."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def alignment_score(z: float, direction: TradeDirection, hmm_state: int) -> float:
    """Pontua alinhamento Z com direcao do cluster; HMM tendencia inverte sinal desejado."""
    if direction == TradeDirection.CALL:
        raw = max(0.0, z if int(hmm_state) == 1 else -z)
    elif direction == TradeDirection.PUT:
        raw = max(0.0, -z if int(hmm_state) == 1 else z)
    else:
        return 0.0
    if hmm_state == 1:
        return raw * 0.5
    return raw


def wr_blend_score(sym: str, wr_scores: dict[str, float] | None, weight: float) -> float:
    """Contribuicao do win-rate rolling ao score composto do indice."""
    if not wr_scores or weight <= 0.0:
        return 0.0
    raw = wr_scores.get(sym)
    if raw is None:
        return 0.0
    return max(0.0, float(raw)) * weight
