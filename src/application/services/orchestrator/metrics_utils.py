"""Metricas padrao para fluxo simples e estado neutro do orquestrador."""

from src.domain.models.trade import TradeDirection


def neutral_metrics() -> dict:
    """Retorna metrica neutra para estado sem direcao."""
    return {
        "conviction": 0.0,
        "direction": "WAIT",
        "macro_bias": 0.0,
        "pattern_tags": [],
        "price": 0.0,
        "call_score": 0.0,
        "put_score": 0.0,
        "prob_call": 0.5,
        "prob_put": 0.5,
        "h1_trend": 0.0,
        "d1_trend": 0.0,
        "macro_slope": 0.0,
        "mtf_structure_bull_n": 0,
        "mtf_structure_bear_n": 0,
    }


def stub_metrics(direction: TradeDirection) -> dict:
    """Retorna metrica sintetica para modo simples CALL/PUT."""
    pc = 1.0 if direction == TradeDirection.CALL else 0.0
    return {
        "conviction": 1.0,
        "direction": direction.name,
        "macro_bias": 1.0 if direction == TradeDirection.CALL else -1.0,
        "pattern_tags": [],
        "price": 0.0,
        "call_score": 1.0 if direction == TradeDirection.CALL else 0.0,
        "put_score": 1.0 if direction == TradeDirection.PUT else 0.0,
        "prob_call": pc,
        "prob_put": 1.0 - pc,
        "h1_trend": 0.0,
        "d1_trend": 0.0,
        "macro_slope": 0.0,
        "mtf_structure_bull_n": 0,
        "mtf_structure_bear_n": 0,
    }
