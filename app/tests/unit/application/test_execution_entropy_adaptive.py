"""Testes de entropia adaptativa no resolver."""

from src.application.services.execution_entropy_adaptive import resolve_dl_entropy_penalty
from src.domain.math.probability_entropy import adaptive_entropy_ceiling


def test_adaptive_entropy_ceiling_tightens_with_regime():
    base = adaptive_entropy_ceiling(0.92, 0.0, squeeze_tighten=0.35, entropy_floor=0.0)
    tight = adaptive_entropy_ceiling(0.92, 1.0, squeeze_tighten=0.35, entropy_floor=0.0)
    assert base == 0.92
    assert tight < base
    assert tight >= 0.0


def test_resolve_dl_entropy_penalty_uses_regime_from_metrics():
    metrics = {"volatility_regime": 0.8, "calibrated_prob": 0.55}
    penalty, ent, ceiling = resolve_dl_entropy_penalty(
        0.55,
        metrics,
        calibration_cfg={"entropy_ceiling": 0.92, "entropy_regime_tighten": 0.35},
    )
    assert ent > 0.0
    assert ceiling < 0.92
    assert 0.0 <= penalty <= 1.0


def test_resolve_dl_entropy_penalty_high_confidence_low_penalty():
    metrics = {"volatility_regime": 0.0}
    penalty, _, _ = resolve_dl_entropy_penalty(
        0.95,
        metrics,
        calibration_cfg={"entropy_ceiling": 0.92, "entropy_floor": 0.0},
    )
    assert penalty < 0.25


def test_resolve_dl_entropy_penalty_recalculates_regime_from_indicators():
    metrics = {
        "indicators": {
            "bb_width": 0.02,
            "atr_norm": 0.01,
            "adx": 0.3,
            "vol_ratio": 1.1,
        }
    }
    penalty, _, ceiling = resolve_dl_entropy_penalty(
        0.55,
        metrics,
        calibration_cfg={"entropy_ceiling": 0.92, "entropy_regime_tighten": 0.35},
        dynamic_cfg={"baseline_lookback": 8},
    )
    assert 0.0 <= penalty <= 1.0
    assert ceiling <= 0.92
