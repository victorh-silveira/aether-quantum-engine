from types import SimpleNamespace

import pytest

from src.application.services.execution_direction_resolver import (
    _strict_anchor_direction,
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
        "predicted_payoff_edge": 0.06,
        "meta_classifier_applied": True,
    }
    if calibrated_prob is not None:
        metrics["calibrated_prob"] = calibrated_prob
    return {"direction": direction, "metrics": metrics}


def _c0015_entry():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = -0.22
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.03}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": -0.02}
    return entry


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
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["direction_inverted"] is False
    assert metrics["trade_score"] == 0.82
    assert metrics["exec_direction"] == "CALL"
    assert metrics["dl_direction"] == "CALL"
    assert metrics["direction_margin"] == pytest.approx(0.32)


def test_resolve_follows_dl_put_and_scores_complement():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.30)
    result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["trade_score"] == 0.70
    assert metrics["resolved_direction"] == "PUT"
    assert metrics["direction_margin"] == pytest.approx(0.20)


def test_resolve_gray_zone_raw_prob_blocks_call_on_bull_with_meta():
    entry = _entry(direction=None, raw_prob=0.51)
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["predicted_payoff_edge"] = 0.05
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is None


def test_resolve_gray_zone_raw_prob_allows_call_on_bull_without_meta():
    entry = _entry(direction=None, raw_prob=0.62)
    entry["metrics"].pop("predicted_payoff_edge", None)
    entry["metrics"].pop("meta_classifier_applied", None)
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_rejects_weak_margin_without_meta():
    entry = _entry(direction=None, raw_prob=0.51)
    entry["metrics"].pop("predicted_payoff_edge", None)
    entry["metrics"].pop("meta_classifier_applied", None)
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is None
    assert entry["metrics"]["quality_guard_reject"] is True


def test_resolve_returns_none_without_prob():
    entry = {"direction": None, "metrics": {"deploy_ok": True, "execute": True}}
    assert resolve_execution_direction(entry) is None


def test_resolve_returns_none_when_technically_blocked():
    assert resolve_execution_direction(_entry(execute=False, gate_reason="data")) is None


def test_resolve_defaults_prob_when_direction_without_prob():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.6, "predicted_payoff_edge": 0.06},
    }
    result = resolve_execution_direction(
        entry,
        symbol="RDBEAR",
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            }
        },
    )
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


def test_resolve_applies_prefetched_positive_edge_with_organic_tcn_score():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.14
    entry["metrics"]["meta_classifier_applied"] = True
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": True}},
        symbol="RDBULL",
    )
    assert result is not None
    assert result[1]["trade_score"] == pytest.approx(0.70)
    assert result[1]["conviction"] == pytest.approx(0.70)


def test_strict_anchor_rejects_negative_edge():
    assert (
        _strict_anchor_direction(
            0.70,
            -0.05,
            "RDBULL",
            meta_applied=True,
            call_conviction_ok=True,
            put_book_ok=True,
        )
        is None
    )


def test_resolve_mild_negative_edge_blocked_by_quality_gate():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = -0.08
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.09}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is None
    assert entry["metrics"]["quality_guard_reject"] is True


def test_resolve_meta_disabled_keeps_tcn_score_when_edge_strong():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.06
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": False}},
        symbol="RDBULL",
    )
    assert result is not None
    assert result[1]["trade_score"] == pytest.approx(0.70)


def test_resolve_rejects_weak_edge_without_meta_prefetch():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.01
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=2,
        pending_loss={},
        pending_loss_total=lambda: 0.0,
    )
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": True}},
        symbol="RDBULL",
        risk_manager=risk_manager,
    )
    assert result is None
    assert entry["metrics"]["quality_guard_reject"] is True


def test_resolve_c0015_negative_edge_blocked_by_quality_gate(caplog):
    entry = _c0015_entry()
    with caplog.at_level("INFO"):
        result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is None
    assert entry["metrics"]["quality_guard_reject"] is True
