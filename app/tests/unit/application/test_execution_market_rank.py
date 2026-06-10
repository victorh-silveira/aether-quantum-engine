from src.application.services.execution_market_rank import (
    _binary_alignment_bonus,
    _candle_implied_direction,
    _flow_overrides_dl,
    _recovery_score_adjustment,
    _resolve_mandatory_weak_direction,
    _weak_signal_multiplier,
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_mean_reversion_from_binary_ctx():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.51, "binary_ctx": {"sma_z": 0.005}},
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_keeps_strong_raw_side():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.62, "binary_ctx": {"sma_z": 0.005}},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_oversold_reversal_call():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"raw_prob": 0.51, "binary_ctx": {"sma_z": -0.005}},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_binary_alignment_bonus_put_branch():
    ctx = {
        "rsi": 0.60,
        "sma_z": 0.002,
        "close_loc": 0.40,
        "variance_ratio": 0.70,
        "rel_vol": 0.30,
        "z_spread": 0.10,
    }
    assert _binary_alignment_bonus(TradeDirection.PUT, ctx) > 0.05


def test_market_decision_score_uses_all_dl_indicators():
    metrics = {
        "trade_score": 0.56,
        "raw_prob": 0.56,
        "val_accuracy": 0.55,
        "edge": 0.06,
        "execute": True,
        "deploy_ok": True,
        "live_win_rate": 0.55,
        "val_brier": 0.30,
        "binary_ctx": {"rsi": 0.40, "sma_z": -0.001, "close_loc": 0.55, "variance_ratio": 0.8, "rel_vol": 0.3},
    }
    score = market_decision_score(
        metrics,
        exec_direction=TradeDirection.CALL,
        recovery_active=True,
        symbol="R_50",
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert score > 0.5


def test_market_decision_score_penalizes_weak_trade_and_raw():
    weak = market_decision_score(
        {"trade_score": 0.0, "raw_prob": 0.51, "val_accuracy": 0.71},
        exec_direction=TradeDirection.CALL,
    )
    strong = market_decision_score(
        {"trade_score": 0.56, "raw_prob": 0.56, "val_accuracy": 0.0, "deploy_ok": False},
        exec_direction=TradeDirection.CALL,
        symbol="R_50",
    )
    assert strong > weak


def test_mandatory_pool_eligible_rejects_data_gate():
    entry = {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "data"}}
    assert mandatory_pool_eligible(entry) is False
    assert mandatory_pool_eligible({"direction": TradeDirection.PUT, "metrics": {"raw_prob": 0.44}}) is True


def test_mandatory_pool_eligible_rejects_training_cooldown_and_pause():
    for gate in ("training", "cooldown", "session_pause"):
        entry = {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": gate, "raw_prob": 0.62, "trade_score": 0.70},
        }
        assert mandatory_pool_eligible(entry) is False


def test_mandatory_pool_eligible_rejects_deploy_not_ok():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": False, "raw_prob": 0.62, "trade_score": 0.70},
    }
    assert mandatory_pool_eligible(entry) is False


def test_build_market_execution_candidate_marks_inversion():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.51, "binary_ctx": {"sma_z": 0.005}},
    }
    built = build_market_execution_candidate("R_50", entry)
    assert built is not None
    assert built[1] == TradeDirection.PUT
    assert built[2]["direction_inverted"] is True


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


def test_resolve_market_direction_moderate_flow_overrides_dl():
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
    assert resolve_market_direction(entry) == TradeDirection.PUT


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
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.40, strong_metrics) is False
    weak_metrics = {"trade_score": 0.0, "raw_prob": 0.51, "gate_reason": "conviction"}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.20, weak_metrics) is True
    mid_metrics = {"trade_score": 0.58, "raw_prob": 0.58, "execute": True, "deploy_ok": True}
    assert _flow_overrides_dl(TradeDirection.CALL, TradeDirection.PUT, 0.35, mid_metrics) is True


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
