"""Metricas padrao para estado neutro do orquestrador."""


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
