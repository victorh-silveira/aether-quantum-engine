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
    bear_put_prob_expanding,
    bull_call_prob_expanding,
    peer_payoff_edge,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_tracker():
    reset_direction_persistence_tracker()
    yield
    reset_direction_persistence_tracker()


def test_peer_payoff_edge_fallback_paths():
    assert peer_payoff_edge({"metrics": {"predicted_payoff_edge": 0.08}}, {}) == pytest.approx(0.08)
    assert peer_payoff_edge({"metrics": {}}, {"predicted_payoff_edge": 0.06}) == pytest.approx(0.06)
    assert peer_payoff_edge({"metrics": {}}, {}) == pytest.approx(0.0)


def test_bull_call_prob_expanding_false_without_bull_entry():
    assert bull_call_prob_expanding({"metrics": {"calibrated_prob": 0.30}}, None, {}, None) is False


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
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
            "calibrated_prob": 0.20,
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


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_guard_congestion_freeze_on_peer_flip_attempt():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    congested = {
        "metrics": {
            "calibrated_prob": 0.30,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        }
    }
    peer = {"metrics": {"calibrated_prob": 0.55}}
    metrics = dict(congested["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.PUT,
            TradeDirection.PUT,
            metrics,
            entry=congested,
            peer_entry=peer,
            cycle_id=22,
            infra_cfg=None,
        )
        is None
    )


def test_bear_put_prob_expanding_helpers():
    metrics = {"cross_symbol_features": {"cross_symbol_prob_delta": 0.09}, "cross_symbol_prob_delta_mean": 0.02}
    assert bear_put_prob_expanding(
        {"metrics": {"calibrated_prob": 0.40}}, {"metrics": {"calibrated_prob": 0.35}}, metrics, None
    )
    assert not bear_put_prob_expanding({"metrics": {"calibrated_prob": 0.40}}, None, {}, None)


@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BULL", "RDBULL")
@patch("src.application.services.direction_persistence_guard_part2.ANCHOR_BEAR", "RDBEAR")
def test_bear_put_lock_congestion_freeze_before_bull_flip():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    congested = {
        "metrics": {
            "calibrated_prob": 0.65,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        }
    }
    metrics = dict(congested["metrics"])
    assert (
        evaluate_direction_persistence_guard(
            "RDBULL",
            TradeDirection.CALL,
            TradeDirection.CALL,
            metrics,
            entry=congested,
            peer_entry={"metrics": {"calibrated_prob": 0.35}},
            cycle_id=32,
            infra_cfg=None,
        )
        is None
    )


def test_single_symbol_guard_blocks_repeat_without_peer():
    record_direction_outcome("R_10", "CALL", won=False)
    record_direction_outcome("R_10", "CALL", won=False)
    assert (
        evaluate_direction_persistence_guard(
            "R_10",
            TradeDirection.CALL,
            TradeDirection.CALL,
            {},
            entry={"metrics": {"calibrated_prob": 0.7}},
            peer_entry=None,
            cycle_id=40,
            infra_cfg=None,
        )
        is None
    )
