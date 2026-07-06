import pytest

from src.application.services.direction_loss_tracker import (
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.application.services.direction_persistence_guard import (
    _peer_payoff_edge,
    bear_put_prob_expanding,
    bull_call_prob_expanding,
    evaluate_direction_persistence_guard,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_tracker():
    reset_direction_persistence_tracker()
    yield
    reset_direction_persistence_tracker()


def test_peer_payoff_edge_fallback_paths():
    assert _peer_payoff_edge({"metrics": {"predicted_payoff_edge": 0.08}}, {}) == pytest.approx(0.08)
    assert _peer_payoff_edge({"metrics": {}}, {"predicted_payoff_edge": 0.06}) == pytest.approx(0.06)
    assert _peer_payoff_edge({"metrics": {}}, {}) == pytest.approx(0.0)


def test_bull_call_prob_expanding_false_without_bull_entry():
    assert bull_call_prob_expanding({"metrics": {"calibrated_prob": 0.30}}, None, {}, None) is False


def test_guard_freeze_paths_for_failed_peer_flips():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    bull = {
        "metrics": {
            "calibrated_prob": 0.80,
            "edge_zscore": 0.40,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.0},
        }
    }
    bear = {
        "metrics": {
            "calibrated_prob": 0.75,
            "edge_zscore": 0.40,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.0},
        }
    }
    metrics = dict(bear["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.PUT,
            TradeDirection.PUT,
            metrics,
            entry=bear,
            peer_entry=bull,
            cycle_id=20,
            infra_cfg=None,
        )
        is None
    )
    reset_direction_persistence_tracker()
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    bull = {
        "metrics": {
            "calibrated_prob": 0.15,
            "edge_zscore": 0.40,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.0},
        }
    }
    bear = {
        "metrics": {
            "calibrated_prob": 0.75,
            "edge_zscore": 0.40,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.0},
        }
    }
    metrics = dict(bull["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBULL",
            TradeDirection.CALL,
            TradeDirection.CALL,
            metrics,
            entry=bull,
            peer_entry=bear,
            cycle_id=21,
            infra_cfg=None,
        )
        is None
    )


def test_guard_congestion_freeze_on_peer_flip_attempt():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    bull = {
        "metrics": {
            "calibrated_prob": 0.55,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        }
    }
    bear = {
        "metrics": {
            "calibrated_prob": 0.30,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        }
    }
    metrics = dict(bear["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.PUT,
            TradeDirection.PUT,
            metrics,
            entry=bear,
            peer_entry=bull,
            cycle_id=22,
            infra_cfg=None,
        )
        is None
    )


def test_guard_blocks_repeat_put_and_returns_proposed_for_other_direction():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    assert (
        evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.PUT,
            TradeDirection.PUT,
            {"edge_zscore": 0.55, "flow_features": {"micro_tick_acceleration": 0.02}},
            entry={
                "metrics": {
                    "calibrated_prob": 0.35,
                    "edge_zscore": 0.55,
                    "flow_features": {"micro_tick_acceleration": 0.02},
                }
            },
            peer_entry={"metrics": {"calibrated_prob": 0.55}},
            cycle_id=30,
            infra_cfg=None,
        )
        is None
    )
    assert (
        evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.CALL,
            TradeDirection.CALL,
            {},
            entry={
                "metrics": {
                    "calibrated_prob": 0.65,
                    "edge_zscore": 0.55,
                    "flow_features": {"micro_tick_acceleration": 0.02},
                }
            },
            peer_entry={"metrics": {"calibrated_prob": 0.35}},
            cycle_id=31,
            infra_cfg=None,
        )
        == TradeDirection.CALL
    )


def test_bear_put_lock_congestion_freeze_before_bull_flip():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    bull = {
        "metrics": {
            "calibrated_prob": 0.65,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        }
    }
    bear = {"metrics": {"calibrated_prob": 0.35}}
    metrics = dict(bull["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBULL",
            TradeDirection.CALL,
            TradeDirection.CALL,
            metrics,
            entry=bull,
            peer_entry=bear,
            cycle_id=32,
            infra_cfg=None,
        )
        is None
    )


def test_guard_blocks_locked_put_on_rdbull_generic_path():
    record_direction_outcome("RDBULL", "PUT", won=False)
    record_direction_outcome("RDBULL", "PUT", won=False)
    metrics = {"edge_zscore": 0.55, "flow_features": {"micro_tick_acceleration": 0.02}}
    assert (
        evaluate_direction_persistence_guard(
            "RDBULL",
            TradeDirection.PUT,
            TradeDirection.PUT,
            metrics,
            entry={
                "metrics": {
                    "calibrated_prob": 0.35,
                    "edge_zscore": 0.55,
                    "flow_features": {"micro_tick_acceleration": 0.02},
                }
            },
            peer_entry={"metrics": {"calibrated_prob": 0.55}},
            cycle_id=33,
            infra_cfg=None,
        )
        is None
    )


def test_bear_put_prob_expanding_reads_stored_delta_mean():
    metrics = {"cross_symbol_prob_delta_mean": 0.03, "cross_symbol_features": {"cross_symbol_prob_delta": 0.04}}
    assert (
        bear_put_prob_expanding(
            {"metrics": {"calibrated_prob": 0.50}}, {"metrics": {"calibrated_prob": 0.50}}, metrics, None
        )
        is True
    )
