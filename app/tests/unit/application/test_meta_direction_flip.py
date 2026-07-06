import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_HARMONIC_WINDOW,
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.direction_loss_tracker import (
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_direction_flip import (
    CHOP_CONGESTION_Z_EDGE,
    META_FLIP_PAYOFF_THRESHOLD_BASE,
    META_FLIP_PAYOFF_THRESHOLD_SQUEEZE,
    META_FLIP_SQUEEZE_TRADE_SCORE,
    META_FLIP_TRADE_SCORE,
    REGIME_CHOP_CONGESTION,
    SIGNAL_SUSPENDED,
    TICK_ACCEL_NEUTRAL_EPS,
    apply_meta_direction_flip,
    apply_regime_freeze_if_congested,
    chop_congestion_regime_active,
    flipped_direction,
    log_d_squeeze_audit,
    micro_volatility_squeeze_active,
    resolve_dynamic_flip_threshold,
    should_flip_direction,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_bb_buffer():
    reset_bb_width_buffer()
    reset_direction_persistence_tracker()
    yield
    reset_bb_width_buffer()
    reset_direction_persistence_tracker()


def _prime_bb(value: float, count: int = BB_WIDTH_HARMONIC_WINDOW) -> None:
    for _ in range(count):
        record_bb_width(value)


def test_should_flip_when_payoff_below_threshold_and_meta_applied():
    assert should_flip_direction(TradeDirection.CALL, 0.35, meta_applied=True)
    assert not should_flip_direction(TradeDirection.CALL, 0.55, meta_applied=True)
    assert not should_flip_direction(TradeDirection.PUT, 0.30, meta_applied=False)


def test_should_flip_respects_elevated_squeeze_threshold():
    assert should_flip_direction(
        TradeDirection.CALL,
        0.46,
        meta_applied=True,
        flip_threshold=META_FLIP_PAYOFF_THRESHOLD_SQUEEZE,
    )
    assert not should_flip_direction(
        TradeDirection.CALL,
        0.50,
        meta_applied=True,
        flip_threshold=META_FLIP_PAYOFF_THRESHOLD_SQUEEZE,
    )


def test_flipped_direction_opposes_tcn():
    assert flipped_direction(TradeDirection.CALL) == TradeDirection.PUT
    assert flipped_direction(TradeDirection.PUT) == TradeDirection.CALL


def test_micro_volatility_squeeze_active_bb_width_and_negative_accel():
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"indicators": {"bb_width": 0.020}}) is True
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"micro_indicators": {"bb_width": 0.020}}) is True
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"macro_indicators": {"bb_width": 0.020}}) is True
    assert micro_volatility_squeeze_active({"flow_features": {"micro_tick_acceleration": -0.02}}) is True
    assert (
        micro_volatility_squeeze_active(
            {
                "indicators": {"bb_width": 0.09},
                "flow_features": {"micro_tick_acceleration": 0.04},
            }
        )
        is False
    )
    _prime_bb(0.035)
    assert micro_volatility_squeeze_active({"indicators": {"bb_width": 0.035}}) is False


def test_resolve_dynamic_flip_threshold_elevates_on_squeeze():
    _prime_bb(0.050)
    metrics = {"indicators": {"bb_width": 0.020}}
    threshold, squeeze = resolve_dynamic_flip_threshold(metrics)
    assert squeeze is True
    assert threshold == pytest.approx(META_FLIP_PAYOFF_THRESHOLD_SQUEEZE)


def test_resolve_dynamic_flip_threshold_base_without_squeeze():
    metrics = {
        "indicators": {"bb_width": 0.09},
        "flow_features": {"micro_tick_acceleration": 0.02},
    }
    threshold, squeeze = resolve_dynamic_flip_threshold(metrics)
    assert squeeze is False
    assert threshold == pytest.approx(META_FLIP_PAYOFF_THRESHOLD_BASE)


def test_apply_meta_direction_flip_call_to_put():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.CALL,
        metrics,
        0.38,
        meta_applied=True,
        tcn_probability=0.68,
    )
    assert exec_dir == TradeDirection.PUT
    assert score == pytest.approx(META_FLIP_TRADE_SCORE)
    assert metrics["meta_direction_flip"] is True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["dl_direction"] == "CALL"
    assert metrics["trade_score"] == pytest.approx(META_FLIP_TRADE_SCORE)


def test_apply_meta_direction_flip_put_to_call():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.PUT,
        metrics,
        0.30,
        meta_applied=True,
        tcn_probability=0.32,
    )
    assert exec_dir == TradeDirection.CALL
    assert metrics["meta_direction_flip"] is True
    assert metrics["direction_inverted"] is True
    assert score == pytest.approx(META_FLIP_TRADE_SCORE)


def test_apply_meta_direction_flip_skips_when_payoff_healthy():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.CALL,
        metrics,
        0.62,
        meta_applied=True,
        tcn_probability=0.62,
    )
    assert exec_dir == TradeDirection.CALL
    assert score == pytest.approx(0.62)
    assert metrics.get("meta_direction_flip") is not True


def test_apply_meta_direction_flip_c0015_squeeze_defensive_score():
    _prime_bb(0.050)
    metrics = {
        "indicators": {"bb_width": 0.020},
        "flow_features": {"micro_tick_acceleration": 0.01},
    }
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.CALL,
        metrics,
        0.46,
        meta_applied=True,
        tcn_probability=0.70,
    )
    assert exec_dir == TradeDirection.PUT
    assert metrics["dynamic_flip_threshold"] == pytest.approx(0.49)
    assert metrics["meta_squeeze_active"] is True
    assert metrics["meta_squeeze_flip"] is True
    assert score == pytest.approx(META_FLIP_SQUEEZE_TRADE_SCORE)
    assert metrics["trade_score"] == pytest.approx(0.52)


def test_log_d_squeeze_audit_emits_tag(caplog):
    _prime_bb(0.050)
    metrics = {
        "indicators": {"bb_width": 0.020},
        "flow_features": {"micro_tick_acceleration": 0.01},
        "meta_calibrated_payoff_score": 0.46,
        "dynamic_flip_threshold": 0.49,
        "meta_direction_flip": True,
        "trade_score": 0.52,
    }
    with caplog.at_level("INFO", logger="AETH"):
        log_d_squeeze_audit("RDBULL", metrics)
    assert any("[D-SQUEEZE]" in record.message for record in caplog.records)


def test_regime_freeze_constants():
    assert pytest.approx(0.20) == CHOP_CONGESTION_Z_EDGE
    assert pytest.approx(0.01) == TICK_ACCEL_NEUTRAL_EPS


def test_chop_congestion_inactive_without_persistence_filter():
    metrics = {"edge_zscore": 0.05, "flow_features": {"micro_tick_acceleration": 0.0}}
    assert chop_congestion_regime_active(metrics, persistence_filter_active=False) is False


def test_resolver_blocks_repeat_call_after_two_bull_call_losses():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.70,
            "predicted_payoff_edge": 0.12,
            "meta_classifier_applied": True,
            "edge_zscore": 0.55,
            "flow_features": {"micro_tick_acceleration": 0.03},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        },
    }
    peer = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.55,
            "predicted_payoff_edge": -0.05,
            "edge_zscore": 0.05,
            "flow_features": {"micro_tick_acceleration": 0.0},
        },
    }
    result = resolve_execution_direction(
        entry,
        symbol="RDBULL",
        peer_entry=peer,
        cycle_id=8,
    )
    assert result is None


def test_resolver_flips_bear_put_after_bull_call_streak(caplog):
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    bull = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.55,
            "predicted_payoff_edge": 0.10,
            "meta_classifier_applied": True,
            "edge_zscore": 0.40,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        },
    }
    bear = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.30,
            "predicted_payoff_edge": 0.14,
            "meta_classifier_applied": True,
            "edge_zscore": 0.45,
            "flow_features": {"micro_tick_acceleration": 0.02},
            "cross_symbol_features": {"cross_symbol_prob_delta": 0.08},
        },
    }
    with caplog.at_level("INFO", logger="AETH"):
        result = resolve_execution_direction(
            bear,
            symbol="RDBEAR",
            peer_entry=bull,
            cycle_id=9,
        )
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["anti_trend_lock_flip"] is True
    assert any("REGIME_GUARD" in record.message and "FLIP to PUT" in record.message for record in caplog.records)


def test_regime_freeze_skips_cycle_on_chop_congestion():
    metrics = {
        "edge_zscore": 0.05,
        "flow_features": {"micro_tick_acceleration": 0.0},
    }
    assert apply_regime_freeze_if_congested(metrics, persistence_filter_active=True) is True
    assert metrics["regime_classification"] == REGIME_CHOP_CONGESTION
    assert metrics["signal_status"] == SIGNAL_SUSPENDED
