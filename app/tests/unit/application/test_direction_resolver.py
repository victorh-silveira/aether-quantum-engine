from src.application.services.execution_direction_resolver import (
    _dl_call_put_scores,
    _low_val_accuracy_bias,
    _scoring_weights,
    is_technically_blocked,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection


def _entry(
    *,
    direction=None,
    raw_prob=0.55,
    execute=True,
    gate_reason=None,
    deploy_ok=True,
    val_accuracy=0.70,
    trend_direction="CALL",
    indicators=None,
):
    return {
        "direction": direction,
        "metrics": {
            "execute": execute,
            "gate_reason": gate_reason,
            "deploy_ok": deploy_ok,
            "raw_prob": raw_prob,
            "trade_score": max(raw_prob, 1.0 - raw_prob),
            "val_accuracy": val_accuracy,
            "trend_direction": trend_direction,
            "call_votes": 4,
            "put_votes": 2,
            "indicators": indicators
            or {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 1.10,
                "rsi": 0.52,
                "keltner": 0.55,
                "cmo": 0.05,
            },
        },
    }


def test_technically_blocked_predict_error():
    entry = _entry(execute=False, gate_reason="predict_error")
    assert is_technically_blocked(entry) is True


def test_technically_blocked_deploy_false():
    entry = _entry(deploy_ok=False)
    assert is_technically_blocked(entry) is True


def test_resolve_keeps_dl_direction_when_aligned():
    entry = _entry(direction=TradeDirection.CALL, raw_prob=0.82)
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["direction_inverted"] is False
    assert metrics["direction_margin"] >= 0.0


def test_resolve_flips_put_to_call_on_exhaustion():
    entry = _entry(
        direction=TradeDirection.PUT,
        raw_prob=0.42,
        trend_direction="CALL",
        indicators={
            "hurst": 0.50,
            "adx": 0.30,
            "vol_ratio": 1.00,
            "rsi": 0.25,
            "keltner": -0.20,
            "cmo": -0.10,
        },
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["direction_inverted"] is True
    assert "exhaustion_flip" in metrics["direction_hints"]


def test_resolve_gray_zone_raw_prob_still_picks_side():
    entry = _entry(direction=None, raw_prob=0.51)
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, _ = result
    assert direction == TradeDirection.CALL


def test_resolve_mean_reversion_on_low_hurst():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.58,
        indicators={
            "hurst": 0.40,
            "adx": 0.20,
            "vol_ratio": 0.90,
            "rsi": 0.60,
            "keltner": 0.65,
            "cmo": 0.02,
        },
        trend_direction="PUT",
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert "mean_reversion" in metrics["direction_hints"]


def test_resolve_returns_none_without_raw_prob():
    entry = {"direction": None, "metrics": {"deploy_ok": True, "execute": True}}
    assert resolve_execution_direction(entry) is None


def test_resolve_custom_direction_scoring_weights():
    entry = _entry(direction=TradeDirection.CALL, raw_prob=0.70)
    result = resolve_execution_direction(
        entry,
        exec_cfg={"direction_scoring": {"dl_raw_weight": 0.60, "trend_weight": 0.20}},
    )
    assert result is not None


def test_resolve_uses_direction_when_raw_prob_missing():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "deploy_ok": True,
            "val_accuracy": 0.60,
            "trend_direction": "PUT",
            "indicators": {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 1.10,
                "rsi": 0.52,
                "keltner": 0.55,
                "cmo": 0.10,
            },
        },
    }
    result = resolve_execution_direction(entry)
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_exhaustion_overbought_bias_put():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.58,
        indicators={
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.74,
            "keltner": 1.20,
            "cmo": -0.12,
        },
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    assert "exhaustion_flip" in result[1]["direction_hints"]
    assert "indicator_regime" in result[1]["direction_hints"]


def test_resolve_skips_trend_in_choppy_regime():
    entry = _entry(
        raw_prob=0.62,
        trend_direction="CALL",
        indicators={
            "hurst": 0.55,
            "adx": 0.20,
            "vol_ratio": 0.80,
            "rsi": 0.52,
            "keltner": 0.55,
            "cmo": 0.0,
        },
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    assert "trend_bias" not in result[1]["direction_hints"]


def test_resolve_recovery_active_trend_put_bonus():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.42, trend_direction="PUT")
    result = resolve_execution_direction(entry, recovery_active=True)
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_returns_none_when_technically_blocked():
    entry = _entry(execute=False, gate_reason="data")
    assert resolve_execution_direction(entry) is None


def test_resolve_mean_reversion_rsi_oversold():
    entry = _entry(
        direction=TradeDirection.PUT,
        raw_prob=0.48,
        indicators={
            "hurst": 0.40,
            "adx": 0.20,
            "vol_ratio": 0.90,
            "rsi": 0.40,
            "keltner": 0.55,
            "cmo": 0.0,
        },
        trend_direction="PUT",
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    assert "mean_reversion" in result[1]["direction_hints"]


def test_resolve_low_val_flip_hint():
    entry = _entry(direction=TradeDirection.PUT, raw_prob=0.42, val_accuracy=0.45, trend_direction="CALL")
    result = resolve_execution_direction(entry)
    assert result is not None
    assert "low_val_flip" in result[1]["direction_hints"]


def test_resolve_ignores_invalid_trend_direction():
    entry = _entry(raw_prob=0.62, trend_direction="INVALID")
    result = resolve_execution_direction(entry)
    assert result is not None


def test_resolve_recovery_call_trend_bonus():
    entry = _entry(direction=TradeDirection.CALL, raw_prob=0.55, trend_direction="CALL")
    result = resolve_execution_direction(entry, recovery_active=True)
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_sets_trade_score_when_missing():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.62,
            "val_accuracy": 0.60,
            "trend_direction": "CALL",
            "indicators": {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 1.10,
                "rsi": 0.52,
                "keltner": 0.55,
                "cmo": 0.05,
            },
        },
    }
    result = resolve_execution_direction(entry)
    assert result is not None
    assert result[1]["trade_score"] is not None


def test_resolve_micro_boundary_downgrades_call_at_volatility_top():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.88,
        indicators={
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.69,
            "keltner": 1.16,
            "cmo": 0.25,
            "bb_pct_b": 0.90,
        },
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics["trade_score"] == 0.55
    assert metrics["micro_boundary_exhaustion"] is True


def test_dl_call_put_scores_without_inferable_direction():
    weights = _scoring_weights({})
    assert _dl_call_put_scores({"direction": None, "metrics": {}}, weights) == (0.5, 0.5)


def test_low_val_accuracy_bias_without_dl_direction():
    weights = _scoring_weights({})
    assert _low_val_accuracy_bias({"direction": None, "metrics": {}}, {"val_accuracy": 0.40}, weights) == (
        0.5,
        0.5,
        None,
    )
