from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.domain.models.trade import TradeDirection


def _entry(prob: float, *, calibrated: bool = True) -> dict:
    metrics = {
        "raw_prob": prob,
        "deploy_ok": True,
        "gate_reason": None,
        "dynamic_call_threshold": 0.53,
        "dynamic_put_threshold": 0.47,
    }
    if calibrated:
        metrics["calibrated_prob"] = prob
    return {"direction": None, "metrics": metrics}


def test_entropy_fallback_picks_lowest_entropy():
    decisions = {
        "R_10": _entry(0.82),
        "R_50": _entry(0.51),
    }
    picked = pick_entropy_fallback_candidate(["R_10", "R_50"], decisions)
    assert picked is not None
    symbol, direction, metrics = picked
    assert symbol == "R_10"
    assert direction in (TradeDirection.CALL, TradeDirection.PUT)
    assert metrics.get("execution_mode") == "EXEC_FALLBACK"
    assert metrics.get("fallback_reason") == "entropy_min"


def test_entropy_fallback_skips_technical_blocks():
    blocked = _entry(0.9)
    blocked["metrics"]["gate_reason"] = "predict_error"
    blocked["metrics"]["deploy_ok"] = False
    decisions = {"R_10": blocked, "R_50": _entry(0.62)}
    picked = pick_entropy_fallback_candidate(["R_10", "R_50"], decisions)
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}
