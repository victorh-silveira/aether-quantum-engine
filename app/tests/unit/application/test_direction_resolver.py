import pytest

from src.application.services.execution_direction_resolver import (
    infer_dl_direction,
    is_technically_blocked,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection


def _entry(*, direction=None, raw_prob=0.55, calibrated_prob=None, execute=True, gate_reason=None, deploy_ok=True):
    metrics = {
        "execute": execute,
        "gate_reason": gate_reason,
        "deploy_ok": deploy_ok,
        "raw_prob": raw_prob,
        "val_accuracy": 0.70,
    }
    if calibrated_prob is not None:
        metrics["calibrated_prob"] = calibrated_prob
    return {"direction": direction, "metrics": metrics}


def test_technically_blocked_predict_error():
    assert is_technically_blocked(_entry(execute=False, gate_reason="predict_error")) is True


def test_technically_blocked_deploy_false():
    assert is_technically_blocked(_entry(deploy_ok=False)) is True


def test_infer_dl_direction_call_when_prob_above_pivot():
    assert infer_dl_direction(_entry(direction=None, raw_prob=0.62)) == TradeDirection.CALL


def test_infer_dl_direction_put_when_prob_below_pivot():
    assert infer_dl_direction(_entry(direction=None, raw_prob=0.40)) == TradeDirection.PUT


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


def test_infer_dl_direction_none_without_prob():
    assert infer_dl_direction({"direction": None, "metrics": {}}) is None


def test_resolve_follows_dl_call_and_scores_calibrated_prob():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.82)
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["direction_inverted"] is False
    assert metrics["trade_score"] == 0.82
    assert metrics["exec_direction"] == "CALL"
    assert metrics["dl_direction"] == "CALL"
    assert metrics["direction_margin"] == 0.82 - 0.18


def test_resolve_follows_dl_put_and_scores_complement():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.30)
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["trade_score"] == 0.70
    assert metrics["resolved_direction"] == "PUT"


def test_resolve_gray_zone_raw_prob_still_picks_call():
    result = resolve_execution_direction(_entry(direction=None, raw_prob=0.51))
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_returns_none_without_prob():
    entry = {"direction": None, "metrics": {"deploy_ok": True, "execute": True}}
    assert resolve_execution_direction(entry) is None


def test_resolve_returns_none_when_technically_blocked():
    assert resolve_execution_direction(_entry(execute=False, gate_reason="data")) is None


def test_resolve_defaults_prob_when_direction_without_prob():
    entry = {"direction": TradeDirection.PUT, "metrics": {"deploy_ok": True, "val_accuracy": 0.6}}
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["trade_score"] == 0.55


def test_resolve_ignores_tactical_config_and_corr_matrix():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    result = resolve_execution_direction(
        entry,
        exec_cfg={"regime_evaluator": {"enabled": True}},
        symbol="RDBULL",
        corr_matrix={("RDBULL", "RDBEAR"): 0.9},
        recovery_active=True,
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1]["direction_inverted"] is False


def test_resolve_applies_prefetched_meta_payoff_score():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["meta_calibrated_payoff_score"] = 0.81
    entry["metrics"]["meta_classifier_applied"] = True
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": True}},
    )
    assert result is not None
    assert result[1]["trade_score"] == pytest.approx(0.81)
    assert result[1]["conviction"] == pytest.approx(0.81)


def test_resolve_flips_call_to_put_when_meta_payoff_saturated():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["meta_calibrated_payoff_score"] = 0.35
    entry["metrics"]["meta_classifier_applied"] = True
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["exec_direction"] == "PUT"
    assert metrics["dl_direction"] == "CALL"
    assert metrics["meta_direction_flip"] is True
    assert metrics["trade_score"] == pytest.approx(0.75)
    assert metrics["direction_inverted"] is True


def test_resolve_flips_put_to_call_when_meta_payoff_saturated():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.30)
    entry["metrics"]["meta_calibrated_payoff_score"] = 0.38
    entry["metrics"]["meta_classifier_applied"] = True
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["meta_direction_flip"] is True
    assert metrics["trade_score"] == pytest.approx(0.75)


def test_resolve_meta_disabled_keeps_tcn_score():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": False}},
    )
    assert result is not None
    assert result[1]["trade_score"] == pytest.approx(0.70)
