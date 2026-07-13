import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
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


def test_resolve_call_keeps_dl_direction_without_tick_accel_gate():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["cross_symbol_features"] = {"cross_symbol_prob_delta": 0.20}
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"cross_symbol_prob_delta_mean": 0.10}},
        symbol="RDBULL",
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_without_symbol_generic_call_path():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    result = resolve_execution_direction(entry)
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_without_symbol_keeps_generic_call_path():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["cross_symbol_features"] = {"cross_symbol_prob_delta": 0.20}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": -0.01}
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"cross_symbol_prob_delta_mean": 0.10}},
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_put_on_bull_follows_dl_direction():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.38, calibrated_prob=0.30)
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_put_on_bear_with_positive_edge():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.38, calibrated_prob=0.30)
    entry["metrics"]["predicted_payoff_edge"] = 0.05
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["cross_symbol_features"] = {"cross_symbol_vol_ratio_diff": -0.1}
    result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_c0015_positive_edge_keeps_organic_score_without_squeeze():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.11
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.09}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["trade_score"] == pytest.approx(0.70)


def test_resolve_without_symbol_uses_generic_put_path():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.38, calibrated_prob=0.30)
    result = resolve_execution_direction(entry)
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_cross_prob_delta_mean_from_metrics():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["cross_symbol_prob_delta_mean"] = 0.12
    entry["metrics"]["cross_symbol_features"] = {"cross_symbol_prob_delta": 0.20}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.03}
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None


def test_resolve_without_prefetch_keeps_organic_score_when_meta_enabled_with_strong_edge():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.08
    entry["metrics"]["meta_classifier_applied"] = False
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": True}},
        symbol="RDBULL",
    )
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["trade_score"] == pytest.approx(0.70)
    assert metrics["predicted_payoff_edge"] == pytest.approx(0.08)
    assert metrics["meta_classifier_applied"] is False
