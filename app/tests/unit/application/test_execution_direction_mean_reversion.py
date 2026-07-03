from src.application.services.execution_direction_mean_reversion import apply_contraction_mean_reversion_flip
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def test_contraction_flip_overbought_puts_against_dl_call():
    metrics = {
        "direction_call_score": 0.62,
        "direction_put_score": 0.41,
        "trade_score": 0.62,
        "indicators": {"rsi": 0.78, "cmo": 0.52, "vol_ratio": 0.72},
    }
    direction, hints = apply_contraction_mean_reversion_flip(
        TradeDirection.CALL,
        TradeDirection.CALL,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.PUT
    assert metrics["mean_reversion_expansion_flip"] is True
    assert metrics["direction_inverted"] is True
    assert "mean_reversion_expansion_flip" in hints


def test_contraction_flip_oversold_puts_against_dl_put():
    metrics = {
        "direction_call_score": 0.41,
        "direction_put_score": 0.62,
        "trade_score": 0.62,
        "indicators": {"rsi": 0.22, "cmo": -0.52, "vol_ratio": 0.72},
    }
    direction, hints = apply_contraction_mean_reversion_flip(
        TradeDirection.PUT,
        TradeDirection.PUT,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL
    assert metrics["mean_reversion_expansion_flip"] is True
    assert "mean_reversion_expansion_flip" in hints


def test_contraction_flip_skips_invalid_indicators():
    metrics = {"indicators": "invalid"}
    direction, hints = apply_contraction_mean_reversion_flip(
        TradeDirection.CALL,
        TradeDirection.CALL,
        ["keep"],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL
    assert hints == ["keep"]


def test_contraction_flip_skipped_when_vol_not_contracting():
    metrics = {"indicators": {"rsi": 0.80, "cmo": 0.50, "vol_ratio": 0.95}}
    direction, _ = apply_contraction_mean_reversion_flip(
        TradeDirection.CALL,
        TradeDirection.CALL,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL


def test_contraction_flip_skipped_when_compression_trap_inverted():
    metrics = {
        "compression_trap_inverted": True,
        "indicators": {"rsi": 0.78, "cmo": 0.52, "vol_ratio": 0.72},
    }
    direction, hints = apply_contraction_mean_reversion_flip(
        TradeDirection.CALL,
        TradeDirection.CALL,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL
    assert hints == []


def test_contraction_flip_oversold_branch():
    metrics = {
        "direction_call_score": 0.41,
        "direction_put_score": 0.62,
        "trade_score": 0.62,
        "indicators": {"rsi": 0.20, "cmo": -0.50, "vol_ratio": 0.72},
    }
    direction, _ = apply_contraction_mean_reversion_flip(
        TradeDirection.PUT,
        TradeDirection.PUT,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL


def test_contraction_flip_no_extremes_keeps_direction():
    metrics = {"indicators": {"rsi": 0.50, "cmo": 0.10, "vol_ratio": 0.72}}
    direction, _ = apply_contraction_mean_reversion_flip(
        TradeDirection.CALL,
        TradeDirection.CALL,
        [],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert direction == TradeDirection.CALL


def test_resolver_mean_reversion_flip_on_exhaustion_contraction():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.58,
            "trade_score": 0.58,
            "val_accuracy": 0.72,
            "trend_direction": None,
            "call_votes": 4,
            "put_votes": 2,
            "indicators": {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 0.75,
                "rsi": 0.74,
                "keltner": 1.10,
                "cmo": 0.53,
            },
        },
    }
    result = resolve_execution_direction(entry, exec_cfg={"exhaustion_gate": {}, "regime_evaluator": {"enabled": True}})
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics.get("universal_regime") == "CLIMAX_EXHAUSTION"


def test_resolver_trend_expansion_preserves_dl_on_high_vol():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.65,
            "trade_score": 0.65,
            "val_accuracy": 0.70,
            "trend_direction": None,
            "call_votes": 4,
            "put_votes": 2,
            "indicators": {
                "hurst": 0.57,
                "adx": 0.30,
                "vol_ratio": 1.18,
                "rsi": 0.68,
                "keltner": 1.20,
                "cmo": 0.05,
            },
        },
    }
    result = resolve_execution_direction(entry, exec_cfg={"exhaustion_gate": {}, "regime_evaluator": {"enabled": True}})
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("universal_regime") == "TREND_EXPANSION"
