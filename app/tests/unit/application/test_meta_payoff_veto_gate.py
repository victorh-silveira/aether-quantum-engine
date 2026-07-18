from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_classifier_stacking import prefetch_meta_payoff_for_decisions
from src.application.services.meta_payoff_veto_gate import (
    META_PAYOFF_NEGATIVE_ZSCORE_VETO,
    apply_meta_payoff_negative_zscore_veto,
    classify_payoff_edge_expectancy,
    is_execution_signal_vetoed,
    meta_payoff_zscore,
    resolve_payoff_edge_expectancy,
    should_veto_meta_payoff_negative_zscore,
    stamp_payoff_edge_expectancy,
)
from src.application.services.payoff_edge_zscore import reset_payoff_edge_buffer
from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import cross_veto_recovery_waiver_allowed
from src.domain.risk.risk_recovery_state import (
    critical_recovery_stress,
    meta_payoff_veto_emergency_waiver,
    tcn_macro_ultra_extreme_conviction,
)


def _risk_manager(*, linear: int = 0, pending: float = 0.0):
    return type(
        "RM",
        (),
        {
            "consecutive_losses_linear": linear,
            "pending_loss": {},
            "pending_loss_total": lambda self, p=pending: float(p),
        },
    )()


def _entry(
    *,
    raw_prob: float,
    calibrated_prob: float | None = None,
    direction: TradeDirection,
    predicted_edge: float = 1.12,
    edge_expectancy: str = "NO_EDGE_NEUTRAL",
):
    prob = calibrated_prob if calibrated_prob is not None else raw_prob
    return {
        "direction": direction,
        "metrics": {
            "deploy_ok": True,
            "execute": True,
            "raw_prob": raw_prob,
            "calibrated_prob": prob,
            "predicted_payoff_edge": predicted_edge,
            "meta_classifier_applied": True,
            "edge_expectancy": edge_expectancy,
            "val_accuracy": 0.72,
            "indicators": {"hurst": 0.62, "bb_width": 0.14, "atr_norm": 0.02},
        },
    }


@pytest.fixture(autouse=True)
def _reset_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


def _stamp_negative_zscore(metrics: dict, z_score: float = -0.77) -> None:
    metrics["meta_payoff_edge_zscore"] = z_score
    metrics["edge_zscore"] = z_score


def test_classify_payoff_edge_expectancy_negative_zscore():
    assert classify_payoff_edge_expectancy(0.12, z_score=-0.77) == "NO_EDGE_NEUTRAL"
    assert classify_payoff_edge_expectancy(0.12, z_score=-0.10) == "WIN_EXPECTED"
    assert classify_payoff_edge_expectancy(-0.02, z_score=0.5) == "LOSS_EXPECTED"
    assert classify_payoff_edge_expectancy(0.02, z_score=0.0) == "NO_EDGE_NEUTRAL"


def test_should_veto_meta_payoff_negative_zscore_skips_neutral_with_low_z():
    metrics = {
        "predicted_payoff_edge": 1.12,
        "edge_expectancy": "NO_EDGE_NEUTRAL",
        "trade_score": 0.80,
    }
    _stamp_negative_zscore(metrics)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.PUT) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["trade_score"] < 0.80
    assert metrics["meta_soft_veto_penalty"] > 0.0


def test_should_veto_meta_payoff_negative_zscore_skips_loss_expected():
    metrics = {
        "predicted_payoff_edge": -0.05,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.70,
    }
    _stamp_negative_zscore(metrics)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is True


def test_should_veto_meta_payoff_negative_zscore_overrides_win_expected():
    metrics = {
        "predicted_payoff_edge": 1.26,
        "edge_expectancy": "WIN_EXPECTED",
        "trade_score": 0.88,
    }
    _stamp_negative_zscore(metrics, z_score=-1.47)
    assert stamp_payoff_edge_expectancy(metrics) == "NO_EDGE_NEUTRAL"
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["signal_status"] == "SOFT_VETO"


def test_resolve_payoff_edge_expectancy_win_expected_with_nonpositive_edge_becomes_loss():
    metrics = {
        "predicted_payoff_edge": 0.0,
        "edge_expectancy": "WIN_EXPECTED",
        "meta_payoff_edge_zscore": -0.55,
        "edge_zscore": -0.55,
    }
    assert stamp_payoff_edge_expectancy(metrics) == "LOSS_EXPECTED"


def test_resolve_payoff_edge_expectancy_defaults_win_when_edge_missing():
    assert resolve_payoff_edge_expectancy({}) == "WIN_EXPECTED"


def test_should_not_veto_win_expected_with_mild_negative_zscore():
    metrics = {
        "predicted_payoff_edge": 0.14,
        "edge_expectancy": "WIN_EXPECTED",
        "meta_payoff_edge_zscore": -0.10,
        "edge_zscore": -0.10,
    }
    assert stamp_payoff_edge_expectancy(metrics) == "WIN_EXPECTED"
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False


def test_is_execution_signal_vetoed_non_dict():
    assert is_execution_signal_vetoed(None) is False


def test_apply_meta_payoff_negative_zscore_veto_forces_skip():
    metrics = {"trade_score": 0.82, "conviction": 0.82}
    apply_meta_payoff_negative_zscore_veto(metrics)
    assert metrics["resolved_direction"] is None
    assert metrics["exec_direction"] is None
    assert metrics["gate_reason"] == META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert metrics["signal_status"] == "SKIP"
    assert metrics["trade_score"] is None


def test_critical_recovery_stress_thresholds():
    assert critical_recovery_stress(3, 80.0) is False
    assert critical_recovery_stress(5, 260.0) is True
    assert critical_recovery_stress(5, 0.0) is False
    assert critical_recovery_stress(0, 260.0) is False


def test_tcn_macro_ultra_extreme_conviction():
    assert tcn_macro_ultra_extreme_conviction(0.18, "PUT") is True
    assert tcn_macro_ultra_extreme_conviction(0.82, "CALL") is True
    assert tcn_macro_ultra_extreme_conviction(0.55, "CALL") is False


def test_meta_payoff_veto_emergency_waiver_requires_stress_and_extreme_tcn():
    metrics = {"raw_prob": 0.18}
    rm = _risk_manager(linear=5, pending=260.0)
    assert meta_payoff_veto_emergency_waiver(metrics, direction="PUT", risk_manager=rm) is True
    assert meta_payoff_veto_emergency_waiver(metrics, direction="PUT", risk_manager=_risk_manager(linear=2)) is False
    assert meta_payoff_veto_emergency_waiver({"raw_prob": 0.55}, direction="CALL", risk_manager=rm) is False


def test_cross_veto_recovery_waiver_allowed_delegates_to_domain():
    metrics = {"raw_prob": 0.82}
    rm = _risk_manager(linear=5, pending=260.0)
    assert cross_veto_recovery_waiver_allowed(metrics, direction="CALL", risk_manager=rm) is True


def test_meta_payoff_zscore_reads_edge_zscore_fallback():
    assert meta_payoff_zscore({"edge_zscore": -0.44}) == pytest.approx(-0.44)
    assert meta_payoff_zscore({}) == 0.0


def test_stamp_payoff_edge_expectancy_derives_from_zscore():
    metrics = {"predicted_payoff_edge": 0.10}
    _stamp_negative_zscore(metrics)
    assert stamp_payoff_edge_expectancy(metrics) == "NO_EDGE_NEUTRAL"


@pytest.mark.asyncio
async def test_prefetch_meta_http_8005_propagates_edge_expectancy():
    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "calibrated_prob": 0.38,
                "feature_vector": [0.1] * 34,
                "micro_indicators": {"rsi": 48.0, "vol_ratio": 1.0},
            },
        },
    }
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    with patch(
        "src.application.services.meta_classifier_stacking.get_meta_classifier_client",
    ) as get_client:
        client = get_client.return_value
        client.predict_meta_batch = AsyncMock(
            return_value=[
                {
                    "predicted_payoff_edge": 1.12,
                    "meta_applied": True,
                    "edge_expectancy": "NO_EDGE_NEUTRAL",
                },
            ],
        )
        get_client.return_value = client
        await prefetch_meta_payoff_for_decisions(decisions, cfg)
    assert decisions["RDBEAR"]["metrics"]["edge_expectancy"] == "NO_EDGE_NEUTRAL"


def test_resolve_execution_direction_skips_on_negative_zscore_veto():
    entry = _entry(raw_prob=0.38, direction=TradeDirection.PUT)
    with patch(
        "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
        side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
    ):
        result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    assert entry["metrics"].get("gate_reason") != META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert result[0] == TradeDirection.PUT


def test_resolve_execution_direction_skips_win_expected_when_zscore_strongly_negative():
    entry = _entry(
        raw_prob=0.83,
        calibrated_prob=0.80,
        direction=TradeDirection.CALL,
        predicted_edge=1.26,
        edge_expectancy="WIN_EXPECTED",
    )
    with patch(
        "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
        side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics, z_score=-1.47),
    ):
        result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert stamp_payoff_edge_expectancy(entry["metrics"]) == "NO_EDGE_NEUTRAL"
    assert entry["metrics"].get("gate_reason") != META_PAYOFF_NEGATIVE_ZSCORE_VETO
    assert result[0] == TradeDirection.CALL


def test_resolve_execution_direction_waives_veto_under_critical_recovery():
    entry = _entry(raw_prob=0.18, direction=TradeDirection.PUT)
    rm = _risk_manager(linear=5, pending=260.0)
    with patch(
        "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
        side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
    ):
        result = resolve_execution_direction(entry, symbol="RDBEAR", risk_manager=rm)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics.get("gate_reason") != META_PAYOFF_NEGATIVE_ZSCORE_VETO


def test_soft_veto_falls_back_to_raw_prob_when_scores_missing():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "raw_prob": 0.81,
    }
    _stamp_negative_zscore(metrics, z_score=-0.90)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["trade_score"] == pytest.approx(0.81 * 0.72)


def test_soft_veto_falls_back_to_resolved_conviction():
    metrics = {
        "predicted_payoff_edge": -0.02,
        "edge_expectancy": "LOSS_EXPECTED",
        "resolved_conviction": 0.77,
        "raw_prob": 0.55,
    }
    _stamp_negative_zscore(metrics, z_score=-0.90)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.PUT) is False
    assert metrics["meta_payoff_soft_veto"] is True
    assert metrics["trade_score"] == pytest.approx(0.77 * 0.72)
