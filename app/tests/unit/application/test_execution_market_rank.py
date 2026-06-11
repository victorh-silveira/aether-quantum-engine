from src.application.services.execution_market_rank import (
    _binary_alignment_bonus,
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
