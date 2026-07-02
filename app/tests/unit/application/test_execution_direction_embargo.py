from src.application.services.execution_direction_resolver import (
    _apply_implied_vol_embargo_weights,
    _atr_norm_peak_anomaly,
    _scoring_weights,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.test_direction_resolver import _entry


def test_atr_norm_peak_anomaly_rejects_invalid_inputs():
    assert not _atr_norm_peak_anomaly({})
    assert not _atr_norm_peak_anomaly({"atr_norm": 0.0})
    assert not _atr_norm_peak_anomaly({"atr_norm": 0.02, "atr_norm_history": [0.01]})
    assert not _atr_norm_peak_anomaly({"atr_norm": 0.02, "atr_norm_history": [0.0, 0.0]})


def test_apply_implied_vol_embargo_skips_without_peak():
    weights = _scoring_weights({})
    metrics = {"indicators": {"implied_vol_ratio": 0.9, "atr_norm": 0.01, "atr_norm_history": [0.01, 0.01]}}
    assert _apply_implied_vol_embargo_weights(metrics, weights) is weights


def test_resolve_applies_implied_vol_embargo_trend_reweight():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.62,
        trend_direction="CALL",
        indicators={
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.52,
            "keltner": 0.55,
            "cmo": 0.05,
            "implied_vol_ratio": 0.85,
            "atr_norm": 0.030,
            "atr_norm_history": [0.010, 0.012, 0.011, 0.013],
        },
    )
    result = resolve_execution_direction(entry)
    assert result is not None
    _, metrics = result
    assert metrics.get("implied_vol_embargo_reweight") is True
