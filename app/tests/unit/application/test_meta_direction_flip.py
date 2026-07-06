import pytest

from src.application.services.meta_direction_flip import (
    META_FLIP_PAYOFF_THRESHOLD_BASE,
    META_FLIP_PAYOFF_THRESHOLD_SQUEEZE,
    META_FLIP_SQUEEZE_TRADE_SCORE,
    META_FLIP_TRADE_SCORE,
    apply_meta_direction_flip,
    flipped_direction,
    log_d_squeeze_audit,
    micro_volatility_squeeze_active,
    resolve_dynamic_flip_threshold,
    should_flip_direction,
)
from src.domain.models.trade import TradeDirection


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
    assert micro_volatility_squeeze_active({"indicators": {"bb_width": 0.03}}) is True
    assert micro_volatility_squeeze_active({"micro_indicators": {"bb_width": 0.04}}) is True
    assert micro_volatility_squeeze_active({"macro_indicators": {"bb_width": 0.05}}) is True
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


def test_resolve_dynamic_flip_threshold_elevates_on_squeeze():
    metrics = {"indicators": {"bb_width": 0.03}}
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
    metrics = {
        "indicators": {"bb_width": 0.03},
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
    metrics = {
        "indicators": {"bb_width": 0.03},
        "flow_features": {"micro_tick_acceleration": 0.01},
        "meta_calibrated_payoff_score": 0.46,
        "dynamic_flip_threshold": 0.49,
        "meta_direction_flip": True,
        "trade_score": 0.52,
    }
    with caplog.at_level("INFO", logger="AETH"):
        log_d_squeeze_audit("RDBULL", metrics)
    assert any("[D-SQUEEZE]" in record.message for record in caplog.records)
