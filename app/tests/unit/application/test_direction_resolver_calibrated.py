from src.application.services.execution_direction_resolver import (
    _dl_call_put_scores,
    _scoring_weights,
    infer_dl_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_uses_calibrated_prob_for_scoring():
    entry = {
        "direction": None,
        "metrics": {
            "raw_prob": 0.58,
            "calibrated_prob": 0.35,
            "val_accuracy": 0.50,
            "trend_direction": None,
            "indicators": {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 1.10,
                "rsi": 0.52,
                "keltner": 0.55,
                "cmo": 0.0,
            },
        },
    }
    call_score, put_score = _dl_call_put_scores(entry, _scoring_weights({}))
    assert put_score > call_score


def test_infer_dl_direction_uses_dynamic_pivot():
    entry = {
        "direction": None,
        "metrics": {
            "calibrated_prob": 0.48,
            "dynamic_call_threshold": 0.56,
            "dynamic_put_threshold": 0.44,
        },
    }
    assert infer_dl_direction(entry) == TradeDirection.PUT
