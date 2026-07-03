from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _climax_entry(*, raw_prob):
    return {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": raw_prob,
            "trade_score": max(raw_prob, 1.0 - raw_prob),
            "val_accuracy": 0.70,
            "trend_direction": "PUT",
            "call_votes": 4,
            "put_votes": 2,
            "indicators": {
                "hurst": 0.55,
                "adx": 0.35,
                "vol_ratio": 1.10,
                "rsi": 0.75,
                "keltner": 0.55,
                "cmo": 0.60,
            },
        },
    }


def test_resolve_dl_conviction_vetoes_tactical_inversion_high_put():
    result = resolve_execution_direction(
        _climax_entry(raw_prob=0.38),
        exec_cfg={"regime_evaluator": {"enabled": True}},
    )
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["dl_inversion_veto"] is True
    assert metrics["direction_inverted"] is False


def test_resolve_allows_tactical_inversion_below_dl_veto_threshold():
    result = resolve_execution_direction(
        _climax_entry(raw_prob=0.42),
        exec_cfg={"regime_evaluator": {"enabled": True}},
    )
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("dl_inversion_veto") is not True
