from unittest.mock import patch

import pytest

from src.application.services.direction_loss_tracker import (
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.application.services.direction_persistence_guard import (
    evaluate_direction_persistence_guard,
)
from src.application.services.direction_persistence_guard_helpers import (
    reset_regime_guard_log_state,
)
from src.application.services.direction_persistence_guard_part2 import (
    _attempt_bear_put_lock_flip,
    _attempt_bull_call_lock_flip,
    _resolve_peer_flip,
)
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_tracker():
    reset_direction_persistence_tracker()
    reset_regime_guard_log_state()
    yield
    reset_direction_persistence_tracker()
    reset_regime_guard_log_state()


def _entry(
    *, prob: float = 0.70, edge: float = 0.12, z_edge: float = 0.55, tick_accel: float = 0.03, delta: float = 0.08
):
    return {
        "metrics": {
            "calibrated_prob": prob,
            "predicted_payoff_edge": edge,
            "edge_zscore": z_edge,
            "flow_features": {"micro_tick_acceleration": tick_accel},
            "cross_symbol_features": {"cross_symbol_prob_delta": delta},
        }
    }


def test_resolve_peer_flip_noop_when_anchors_equal():
    metrics = dict(_entry()["metrics"])
    assert (
        _resolve_peer_flip(
            "R_10",
            TradeDirection.PUT,
            metrics,
            entry=_entry(),
            peer_entry=_entry(prob=0.55),
            cycle_id=1,
            infra_cfg=None,
        )
        is None
    )


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_flips_to_call_on_rdbear_after_bull_put_losses():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    bull = _entry(prob=0.72, edge=0.15, z_edge=0.40)
    bear = _entry(prob=0.30)
    metrics: dict = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=11,
        infra_cfg=None,
    )
    assert result == TradeDirection.CALL
    assert metrics["regime_guard_action"] == "FLIP to CALL"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_flips_to_put_on_rdbull_after_bear_call_losses():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    bear = _entry(prob=0.30)
    bull = _entry(prob=0.72, edge=0.15, z_edge=0.40)
    metrics: dict = dict(bull["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=12,
        infra_cfg=None,
    )
    assert result == TradeDirection.PUT
    assert metrics["regime_guard_action"] == "FLIP to PUT"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_freezes_on_rdbear_after_bull_put_losses_without_expansion():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    bull = _entry(prob=0.40, edge=0.10, z_edge=0.40, delta=0.0)
    bear = _entry(prob=0.55, edge=0.10, z_edge=0.40, delta=0.0)
    metrics = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=13,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_freezes_on_rdbull_after_bear_call_losses_without_expansion():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    bull = _entry(prob=0.55, edge=0.10, z_edge=0.40, delta=0.0)
    bear = _entry(prob=0.70, edge=0.10, z_edge=0.40, delta=0.0)
    metrics = dict(bull["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=14,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_freezes_on_rdbear_after_bull_put_losses_with_negative_edge():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    congested = {
        "metrics": {
            "calibrated_prob": 0.30,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.01},
        }
    }
    metrics = dict(congested["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=congested,
        peer_entry=_entry(prob=0.72),
        cycle_id=15,
        infra_cfg=None,
    )
    assert result is None
    assert (
        metrics.get("signal_status") == SIGNAL_SUSPENDED or metrics.get("regime_guard_action") == "FREEZE: SKIP CYCLE"
    )


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_anti_trend_lock_freezes_on_rdbull_after_bear_call_losses_with_negative_edge():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    congested = {
        "metrics": {
            "calibrated_prob": 0.72,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.01},
        }
    }
    metrics = dict(congested["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=congested,
        peer_entry=_entry(prob=0.30),
        cycle_id=16,
        infra_cfg=None,
    )
    assert result is None
    assert (
        metrics.get("signal_status") == SIGNAL_SUSPENDED or metrics.get("regime_guard_action") == "FREEZE: SKIP CYCLE"
    )


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_bull_call_lock_flip_success_returns_put():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    bull = _entry(prob=0.72, edge=0.15, z_edge=0.40)
    bear = _entry(prob=0.25, edge=0.15, z_edge=0.40)
    metrics = dict(bear["metrics"])
    result = _attempt_bull_call_lock_flip(
        TradeDirection.PUT,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=20,
        infra_cfg=None,
    )
    assert result == TradeDirection.PUT
    assert metrics["regime_guard_action"] == "FLIP to PUT"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_bear_put_lock_flip_success_returns_call():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    bull = _entry(prob=0.72, edge=0.15, z_edge=0.40)
    bear = _entry(prob=0.55, edge=0.15, z_edge=0.40)
    metrics = dict(bull["metrics"])
    result = _attempt_bear_put_lock_flip(
        TradeDirection.CALL,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=21,
        infra_cfg=None,
    )
    assert result == TradeDirection.CALL
    assert metrics["regime_guard_action"] == "FLIP to CALL"


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_resolve_peer_flip_returns_none_for_unmatched_symbol():
    assert (
        _resolve_peer_flip(
            "R_50",
            TradeDirection.CALL,
            dict(_entry()["metrics"]),
            entry=_entry(),
            peer_entry=_entry(prob=0.55),
            cycle_id=22,
            infra_cfg=None,
        )
        is None
    )
