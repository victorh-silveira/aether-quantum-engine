import pytest

from src.application.services.direction_loss_tracker import (
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.application.services.direction_persistence_guard import (
    evaluate_direction_persistence_guard,
    log_regime_guard,
)
from src.application.services.direction_persistence_guard_helpers import (
    _LOGGED_REGIME_GUARD_CYCLES,
    bear_put_prob_expanding,
    bull_call_prob_expanding,
    reset_regime_guard_log_state,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
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


def test_anti_trend_lock_blocks_repeat_call_and_allows_other_direction():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    blocked = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.CALL,
        TradeDirection.CALL,
        {},
        entry=_entry(),
        peer_entry=_entry(prob=0.35),
        cycle_id=7,
        infra_cfg=None,
    )
    assert blocked is None
    reset_direction_persistence_tracker()
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    allowed = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.CALL,
        TradeDirection.CALL,
        {},
        entry=_entry(prob=0.65),
        peer_entry=_entry(prob=0.35),
        cycle_id=5,
        infra_cfg=None,
    )
    assert allowed == TradeDirection.CALL
    repeat_put = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.PUT,
        TradeDirection.PUT,
        {"edge_zscore": 0.55, "flow_features": {"micro_tick_acceleration": 0.02}},
        entry=_entry(prob=0.35),
        peer_entry=_entry(prob=0.55),
        cycle_id=3,
        infra_cfg=None,
    )
    assert repeat_put is None


def test_anti_trend_lock_flips_to_put_on_rdbear_after_bull_call_losses():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    bull = _entry(prob=0.55, edge=0.10)
    bear = _entry(prob=0.30, edge=0.14, z_edge=0.40)
    metrics: dict = dict(bear["metrics"])
    result = evaluate_direction_persistence_guard(
        "RDBEAR",
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        entry=bear,
        peer_entry=bull,
        cycle_id=9,
        infra_cfg=None,
    )
    assert result == TradeDirection.PUT
    assert metrics["regime_guard_action"] == "FLIP to PUT"


def test_anti_trend_lock_flips_to_call_after_bear_put_losses():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
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
        cycle_id=10,
        infra_cfg=None,
    )
    assert result == TradeDirection.CALL
    assert metrics["regime_guard_action"] == "FLIP to CALL"


def test_regime_freeze_when_congestion_or_missing_expansion(caplog):
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    congested = {
        "metrics": {
            "calibrated_prob": 0.55,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.01},
        }
    }
    metrics = dict(congested["metrics"])
    with caplog.at_level("INFO", logger="AETH"):
        frozen = evaluate_direction_persistence_guard(
            "RDBEAR",
            TradeDirection.PUT,
            TradeDirection.PUT,
            metrics,
            entry=congested,
            peer_entry=_entry(prob=0.55),
            cycle_id=9,
            infra_cfg=None,
        )
    assert frozen is None
    assert metrics["signal_status"] == SIGNAL_SUSPENDED
    assert any("FREEZE: SKIP CYCLE" in record.message for record in caplog.records)
    reset_direction_persistence_tracker()
    record_direction_outcome("RDBEAR", "PUT", won=False)
    record_direction_outcome("RDBEAR", "PUT", won=False)
    bull = _entry(prob=0.20, edge=0.10, z_edge=0.40, delta=0.0)
    bear = _entry(prob=0.75, edge=0.10, z_edge=0.40, delta=0.0)
    metrics = dict(bull["metrics"])
    stalled = evaluate_direction_persistence_guard(
        "RDBULL",
        TradeDirection.CALL,
        TradeDirection.CALL,
        metrics,
        entry=bull,
        peer_entry=bear,
        cycle_id=14,
        infra_cfg=None,
    )
    assert stalled is None
    assert metrics["regime_guard_action"] == "FREEZE: SKIP CYCLE"


def test_log_regime_guard_freeze_logs_once_per_cycle(caplog):
    with caplog.at_level("INFO", logger="AETH"):
        log_regime_guard(5, "FREEZE: SKIP CYCLE", 2)
        log_regime_guard(5, "FREEZE: SKIP CYCLE", 2)
    freeze_logs = [record for record in caplog.records if "FREEZE: SKIP CYCLE" in record.message]
    assert len(freeze_logs) == 1


def test_log_regime_guard_prunes_stale_cycle_entries(caplog):
    _LOGGED_REGIME_GUARD_CYCLES[1] = frozenset({"FREEZE: SKIP CYCLE"})
    with caplog.at_level("INFO", logger="AETH"):
        log_regime_guard(150, "FREEZE: SKIP CYCLE", 2)
    assert 1 not in _LOGGED_REGIME_GUARD_CYCLES


def test_prob_expanding_helpers_and_regime_guard_log(caplog):
    metrics: dict = {"cross_symbol_prob_delta_mean": 0.04, "cross_symbol_features": {"cross_symbol_prob_delta": 0.05}}
    assert bear_put_prob_expanding(_entry(prob=0.40), _entry(prob=0.35), metrics, None) is True
    assert bear_put_prob_expanding(_entry(), None, {}, None) is False
    assert bull_call_prob_expanding(_entry(prob=0.30), _entry(prob=0.72), {}, None) is True
    assert bull_call_prob_expanding({"metrics": {"calibrated_prob": 0.80}}, {"metrics": {}}, {}, None) is True
    infra = {"meta_classifier": {"cross_symbol_prob_delta_mean": 0.06}}
    assert bear_put_prob_expanding(_entry(prob=0.50), _entry(prob=0.40), {}, infra) is True
    with caplog.at_level("INFO", logger="AETH"):
        log_regime_guard(6, "FLIP to PUT", 2)
    assert any("[C0006] REGIME_GUARD" in record.message for record in caplog.records)


def test_resolver_and_guard_noop_paths():
    noop = evaluate_direction_persistence_guard(
        None,
        TradeDirection.CALL,
        TradeDirection.CALL,
        {},
        entry=_entry(),
        peer_entry=None,
        cycle_id=1,
        infra_cfg=None,
    )
    assert noop == TradeDirection.CALL
    assert (
        evaluate_direction_persistence_guard(
            "RDBULL",
            TradeDirection.CALL,
            TradeDirection.CALL,
            {},
            entry=_entry(),
            peer_entry=_entry(prob=0.35),
            cycle_id=1,
            infra_cfg=None,
        )
        == TradeDirection.CALL
    )
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    entry = {"direction": TradeDirection.CALL}
    peer = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.55,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
        },
    }
    assert resolve_execution_direction(entry, symbol="RDBULL", peer_entry=peer, cycle_id=8) is None
    assert entry["metrics"].get("signal_status") == SIGNAL_SUSPENDED
