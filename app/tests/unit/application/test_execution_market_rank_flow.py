from src.application.services.execution_market_rank import (
    _recovery_score_adjustment,
    _weak_signal_multiplier,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_strong_dl_keeps_call_despite_bearish_flow():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.54,
            "raw_prob": 0.66,
            "execute": True,
            "deploy_ok": True,
            "binary_ctx": {
                "body": -0.002,
                "body_sum_3": -0.006,
                "close_loc": 0.46,
                "sma_z": 0.001,
                "variance_ratio": 0.90,
                "ema_spread": -0.001,
                "ret_5": -0.002,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_strong_mandatory_without_execute_keeps_dl():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.80,
            "raw_prob": 0.88,
            "execute": False,
            "deploy_ok": True,
            "binary_ctx": {"body": -0.001, "close_loc": 0.46, "sma_z": 0.001},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_borderline_score_keeps_dl_on_weak_flow():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.55,
            "raw_prob": 0.52,
            "execute": True,
            "binary_ctx": {
                "sma_z": 0.005,
                "variance_ratio": 0.75,
                "body": -0.001,
                "body_sum_3": -0.003,
                "close_loc": 0.46,
                "rsi": 0.5,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_uses_flow_when_dl_weak():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "trade_score": 0.0,
            "raw_prob": 0.51,
            "gate_reason": "conviction",
            "binary_ctx": {"body": 0.002, "close_loc": 0.55, "sma_z": 0.0, "variance_ratio": 1.0},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_market_decision_score_penalizes_flow_conflict():
    ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "sma_z": 0.0,
        "rsi_slope": 0.02,
    }
    aligned = market_decision_score(
        {"trade_score": 0.52, "raw_prob": 0.52, "consensus_strength": 0.5, "binary_ctx": ctx},
        exec_direction=TradeDirection.CALL,
    )
    conflict = market_decision_score(
        {"trade_score": 0.52, "raw_prob": 0.52, "consensus_strength": 0.2, "binary_ctx": ctx},
        exec_direction=TradeDirection.PUT,
    )
    assert aligned > conflict


def test_market_decision_score_penalizes_dl_mismatch():
    metrics = {
        "trade_score": 0.56,
        "raw_prob": 0.56,
        "consensus_strength": 0.4,
        "binary_ctx": {},
    }
    aligned = market_decision_score({**metrics, "raw_prob": 0.56}, exec_direction=TradeDirection.CALL)
    mismatch = market_decision_score({**metrics, "raw_prob": 0.56}, exec_direction=TradeDirection.PUT)
    assert aligned > mismatch


def test_market_decision_score_moderate_dl_mismatch_penalty():
    metrics = {
        "trade_score": 0.48,
        "raw_prob": 0.56,
        "consensus_strength": 0.4,
        "binary_ctx": {},
    }
    aligned = market_decision_score(metrics, exec_direction=TradeDirection.CALL)
    mismatch = market_decision_score(metrics, exec_direction=TradeDirection.PUT)
    assert aligned - mismatch == 0.06


def test_weak_signal_multiplier_and_recovery_adjustment():
    ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "rsi_slope": 0.02,
    }
    assert _weak_signal_multiplier(0.40, 0.51, TradeDirection.CALL, ctx) == 0.45
    assert _weak_signal_multiplier(0.52, 0.52, TradeDirection.PUT, ctx) == 0.55
    assert _weak_signal_multiplier(0.60, 0.60, TradeDirection.CALL, ctx) == 1.0
    adjusted = _recovery_score_adjustment(
        0.50,
        recovery_active=True,
        symbol="R_50",
        exec_direction=TradeDirection.PUT,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert adjusted > 0.50
