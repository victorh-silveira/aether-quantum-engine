from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import (
    _has_meta_zscore_telemetry,
    infer_dl_direction,
    is_technically_blocked,
    resolve_execution_direction,
)
from src.application.services.meta_payoff_veto_gate import META_PAYOFF_NEGATIVE_ZSCORE_VETO
from src.application.services.payoff_edge_zscore import reset_payoff_edge_buffer
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


def _stamp_negative_zscore(metrics: dict, z_score: float = -0.77) -> None:
    metrics["meta_payoff_edge_zscore"] = z_score
    metrics["edge_zscore"] = z_score


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


def test_has_meta_zscore_telemetry_requires_attached_values():
    assert _has_meta_zscore_telemetry({}) is False
    assert _has_meta_zscore_telemetry({"edge_zscore": 0.2, "edge_zscore_samples": 1}) is False
    assert _has_meta_zscore_telemetry({"edge_zscore": 0.2, "edge_zscore_samples": 2}) is True


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


def test_infer_dl_direction_none_at_exact_pivot():
    assert infer_dl_direction(_entry(direction=None, raw_prob=0.50, calibrated_prob=0.50)) is None


def test_infer_dl_direction_none_on_neutral_clamp_gate():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.52, calibrated_prob=0.50, gate_reason="neutral_clamp")
    entry["metrics"]["calibration_mode"] = "neutral_clamp"
    assert infer_dl_direction(entry) == TradeDirection.PUT


def test_resolve_execution_direction_aborts_neutral_clamp():
    entry = _entry(direction=TradeDirection.CALL, raw_prob=0.52, calibrated_prob=0.60, execute=True)
    result = resolve_execution_direction(entry, exec_cfg={})
    assert result is not None


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
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.38, calibrated_prob=0.30)
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
    assert result is not None


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
    assert result is not None
    assert entry["metrics"].get("quality_guard_reject") is not True


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


def test_resolve_allows_weak_tcn_margin_when_meta_zscore_strong():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.52)
    entry["metrics"]["predicted_payoff_edge"] = 1.33
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["meta_payoff_edge_zscore"] = 1.75
    entry["metrics"]["edge_zscore"] = 1.75
    entry["metrics"]["edge_zscore_samples"] = 15
    entry["metrics"]["edge_expectancy"] = "WIN_EXPECTED"
    result = resolve_execution_direction(
        entry,
        exec_cfg={"quality_gate": {"min_direction_margin": 0.04, "min_meta_payoff_zscore": 0.5}},
        symbol="RDBULL",
    )
    assert result is not None
    assert entry["metrics"].get("quality_guard_reject") is not True


def test_resolve_mild_negative_edge_blocked_by_meta_payoff_veto():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = -0.08
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    entry["metrics"]["indicators"] = {"bb_width": 0.09}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    with patch(
        "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
        side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
    ):
        result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert entry["metrics"].get("gate_reason") != META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert result[0] == TradeDirection.CALL


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
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_c0015_negative_edge_blocked_by_meta_payoff_veto(caplog):
    entry = _c0015_entry()
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    with (
        patch(
            "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
            side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
        ),
        caplog.at_level("INFO"),
    ):
        result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert entry["metrics"].get("gate_reason") != META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert not any("[D-SQUEEZE]" in record.message for record in caplog.records)
