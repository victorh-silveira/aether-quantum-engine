from src.application.services.execution_market_rank import (
    _candle_implied_direction,
    _flow_overrides_dl,
    _recovery_score_adjustment,
    _resolve_mandatory_weak_direction,
    _strong_dl_signal,
    _weak_signal_multiplier,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_strong_dl_keeps_direction_with_flow_conflict():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.76,
            "raw_prob": 0.88,
            "execute": True,
            "deploy_ok": True,
            "binary_ctx": {"body": -0.001, "close_loc": 0.46, "sma_z": 0.001},
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
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_strong_dl_overrides_on_extreme_flow():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.76,
            "raw_prob": 0.88,
            "execute": True,
            "deploy_ok": True,
            "binary_ctx": {
                "body": -0.003,
                "body_sum_3": -0.009,
                "ema_spread": -0.002,
                "ret_5": -0.003,
                "ret_3": -0.002,
                "close_loc": 0.40,
                "variance_ratio": 0.95,
                "rel_vol": 0.35,
                "body_streak": 3.0,
                "sma_z": 0.0,
                "rsi_slope": -0.02,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_uses_candle_when_flow_weak():
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


def test_resolve_market_direction_moderate_flow_keeps_dl_in_trust_band():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.58,
            "raw_prob": 0.58,
            "execute": True,
            "deploy_ok": True,
            "binary_ctx": {
                "body": -0.002,
                "body_sum_3": -0.006,
                "ema_spread": -0.001,
                "ret_5": -0.002,
                "close_loc": 0.42,
                "variance_ratio": 0.90,
                "rel_vol": 0.30,
                "body_streak": 2.0,
                "sma_z": 0.0,
            },
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
        {"trade_score": 0.52, "raw_prob": 0.52, "binary_ctx": ctx},
        exec_direction=TradeDirection.CALL,
    )
    conflict = market_decision_score(
        {"trade_score": 0.52, "raw_prob": 0.52, "binary_ctx": ctx},
        exec_direction=TradeDirection.PUT,
    )
    assert aligned > conflict


def test_candle_implied_direction_call_and_put():
    assert _candle_implied_direction({"body": 0.002, "close_loc": 0.55}) == TradeDirection.CALL
    assert _candle_implied_direction({"body": -0.002, "close_loc": 0.45}) == TradeDirection.PUT


def test_resolve_mandatory_weak_direction_branches():
    strong_flow_ctx = {
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
    assert _resolve_mandatory_weak_direction(TradeDirection.PUT, strong_flow_ctx) == TradeDirection.CALL
    candle_ctx = {"body": 0.0001, "close_loc": 0.55, "variance_ratio": 1.0}
    assert _resolve_mandatory_weak_direction(TradeDirection.PUT, candle_ctx) == TradeDirection.CALL
    assert _resolve_mandatory_weak_direction(TradeDirection.CALL, {"sma_z": 0.005}) == TradeDirection.PUT


def test_flow_overrides_dl_branches():
    strong_metrics = {"trade_score": 0.76, "raw_prob": 0.88, "execute": True, "deploy_ok": True}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.55, strong_metrics) is True
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.50, strong_metrics) is False
    weak_metrics = {"trade_score": 0.0, "raw_prob": 0.51, "gate_reason": "conviction"}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.25, weak_metrics) is True
    mid_metrics = {"trade_score": 0.52, "raw_prob": 0.52, "execute": True, "deploy_ok": True}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.35, mid_metrics) is True
    border_metrics = {"trade_score": 0.57, "raw_prob": 0.57, "execute": False, "deploy_ok": True}
    assert _flow_overrides_dl(TradeDirection.PUT, TradeDirection.CALL, 0.50, border_metrics) is False
    neutral_metrics = {"trade_score": 0.54, "raw_prob": 0.58, "execute": True, "deploy_ok": True}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.35, neutral_metrics) is True
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.30, neutral_metrics) is False
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.CALL, 0.50, strong_metrics) is False


def test_strong_dl_signal_rejects_deploy_not_ok():
    assert _strong_dl_signal({"trade_score": 0.80, "raw_prob": 0.88, "deploy_ok": False}) is False


def test_market_decision_score_penalizes_inversion_on_strong_signal():
    metrics = {
        "trade_score": 0.80,
        "raw_prob": 0.88,
        "direction_inverted": True,
        "binary_ctx": {},
    }
    base = market_decision_score({**metrics, "direction_inverted": False})
    inverted = market_decision_score(metrics)
    assert base > inverted


def test_market_decision_score_penalizes_moderate_inversion():
    metrics = {
        "trade_score": 0.60,
        "raw_prob": 0.60,
        "direction_inverted": True,
        "binary_ctx": {},
    }
    base = market_decision_score({**metrics, "direction_inverted": False})
    inverted = market_decision_score(metrics)
    assert inverted + 0.08 == base


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
