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


def test_anti_trend_lock_flips_to_call_on_rdbear_after_bull_put_losses():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    bull = _entry(prob=0.30, edge=0.10)
    bear = _entry(prob=0.55, edge=0.14, z_edge=0.40)
    metrics: dict = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=11,
        infra_cfg=None,
    )
    assert result == TradeDirection.CALL
    assert metrics["regime_guard_action"] == "FLIP to CALL"


def test_anti_trend_lock_flips_to_put_on_rdbull_after_bear_call_losses():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    bear = _entry(prob=0.55)
    bull = _entry(prob=0.30, edge=0.15, z_edge=0.40)
    metrics: dict = dict(bull["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=12,
        infra_cfg=None,
    )
    assert result == TradeDirection.PUT
    assert metrics["regime_guard_action"] == "FLIP to PUT"


def test_anti_trend_lock_freezes_on_rdbear_after_bull_put_losses_without_expansion():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    bull = _entry(prob=0.55, edge=0.10)
    bear = _entry(prob=0.30, edge=-0.05, z_edge=0.05, tick_accel=0.0)
    metrics: dict = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=13,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


def test_anti_trend_lock_freezes_on_rdbull_after_bear_call_losses_without_expansion():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    bear = _entry(prob=0.30)
    bull = _entry(prob=0.72, edge=-0.05, z_edge=0.05, tick_accel=0.0)
    metrics: dict = dict(bull["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=14,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


def test_anti_trend_lock_freezes_on_rdbear_after_bull_put_losses_with_negative_edge():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    bull = _entry(prob=0.55, edge=0.10)
    bear = _entry(prob=0.30, edge=-0.05, z_edge=0.40, tick_accel=1.0)
    metrics: dict = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=15,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


def test_anti_trend_lock_freezes_on_rdbull_after_bear_call_losses_with_negative_edge():
    record_direction_outcome("RDBEAR", "CALL", won=False)
    record_direction_outcome("RDBEAR", "CALL", won=False)
    bear = _entry(prob=0.30)
    bull = _entry(prob=0.72, edge=-0.05, z_edge=0.40, tick_accel=1.0)
    metrics: dict = dict(bull["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=16,
        infra_cfg=None,
    )
    assert result is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"
