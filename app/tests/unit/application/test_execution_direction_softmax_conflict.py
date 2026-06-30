from src.application.services.execution_direction_softmax_conflict import resolve_softmax_exhaustion_conflict
from src.domain.models.trade import TradeDirection


def test_softmax_conflict_flips_call_to_put_with_kelly_scale():
    metrics = {
        "exhaustion_conflict": True,
        "exhaustion_penalty": 0.2,
        "indicators": {"rsi": 0.8, "cmo": 0.5, "implied_vol_ratio": 1.2},
        "kelly_fraction_scale": 1.0,
        "direction_hints": [],
    }
    resolved, out = resolve_softmax_exhaustion_conflict(
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        call_score=0.62,
        put_score=0.41,
        exec_cfg={"exhaustion_gate": {"rsi_overbought": 0.73, "cmo_bull": 0.48}},
    )
    assert resolved == TradeDirection.PUT
    assert out.get("execution_mode") == "EXEC_DIVERGENT"
    assert float(out.get("kelly_fraction_scale", 1.0)) < 1.0
