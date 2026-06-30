import pytest

from src.application.services.execution_direction_expansion_veto import apply_expansion_inversion_veto
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _entry(
    *,
    direction=None,
    raw_prob=0.55,
    trend_direction="CALL",
    indicators=None,
):
    return {
        "direction": direction,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": raw_prob,
            "trade_score": max(raw_prob, 1.0 - raw_prob),
            "val_accuracy": 0.70,
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def test_expansion_veto_blocks_inversion_on_high_vol():
    metrics = {
        "direction_call_score": 0.62,
        "direction_put_score": 0.58,
        "trade_score": 0.62,
        "indicators": {"vol_ratio": 1.23, "rsi": 0.20, "keltner": -0.20},
    }
    hints = ["exhaustion_flip"]
    exec_dir, hints_out = apply_expansion_inversion_veto(
        TradeDirection.PUT,
        TradeDirection.CALL,
        hints,
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert exec_dir == TradeDirection.CALL
    assert metrics["direction_inverted"] is False
    assert metrics["expansion_inversion_veto"] is True
    assert metrics["trade_score"] == pytest.approx(0.62 * 0.70, rel=1e-3)
    assert "expansion_veto" in hints_out


def test_expansion_veto_allows_inversion_below_threshold():
    metrics = {
        "direction_call_score": 0.62,
        "direction_put_score": 0.58,
        "trade_score": 0.62,
        "indicators": {"vol_ratio": 1.04},
    }
    exec_dir, _ = apply_expansion_inversion_veto(
        TradeDirection.PUT,
        TradeDirection.CALL,
        ["exhaustion_flip"],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert exec_dir == TradeDirection.PUT
    assert "expansion_inversion_veto" not in metrics


def test_expansion_veto_ignores_low_val_flip():
    metrics = {
        "direction_call_score": 0.55,
        "direction_put_score": 0.60,
        "trade_score": 0.60,
        "indicators": {"vol_ratio": 1.20},
    }
    exec_dir, _ = apply_expansion_inversion_veto(
        TradeDirection.PUT,
        TradeDirection.CALL,
        ["low_val_flip"],
        metrics,
        exec_cfg={},
        clamp01=_clamp01,
    )
    assert exec_dir == TradeDirection.PUT


def test_resolver_expansion_veto_preserves_dl_on_high_vol_exhaustion():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.65,
        trend_direction=None,
        indicators={
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.23,
            "rsi": 0.80,
            "keltner": 1.20,
            "cmo": 0.05,
        },
    )
    result = resolve_execution_direction(entry, exec_cfg={"exhaustion_gate": {}})
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("direction_inverted") is False
    assert metrics.get("expansion_inversion_veto") is True
    assert "expansion_veto" in metrics.get("direction_hints", [])
    pre_veto_strength = max(metrics["direction_call_score"], metrics["direction_put_score"])
    assert metrics["trade_score"] == pytest.approx(pre_veto_strength * 0.70, rel=1e-3)


def test_resolver_expansion_allows_inversion_when_vol_below_threshold():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.65,
        trend_direction=None,
        indicators={
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.04,
            "rsi": 0.80,
            "keltner": 1.20,
            "cmo": 0.05,
        },
    )
    result = resolve_execution_direction(entry, exec_cfg={"exhaustion_gate": {}})
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics.get("direction_inverted") is True
    assert metrics.get("expansion_inversion_veto") is not True
