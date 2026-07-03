from src.application.services.execution_direction_inversion_veto import (
    dl_side_probability,
    veto_inversion_on_dl_conviction,
)
from src.application.services.execution_universal_regime_types import RegimeEvaluation, RegimeState
from src.domain.models.trade import TradeDirection


def test_dl_side_probability_prefers_calibrated_prob():
    metrics = {"calibrated_prob": 0.62, "raw_prob": 0.55}
    assert dl_side_probability(metrics, TradeDirection.CALL) == 0.62
    assert dl_side_probability(metrics, TradeDirection.PUT) == 1.0 - 0.62


def test_dl_side_probability_falls_back_to_raw_prob():
    metrics = {"raw_prob": 0.38}
    assert dl_side_probability(metrics, TradeDirection.PUT) == 1.0 - 0.38


def test_dl_side_probability_falls_back_to_trade_score():
    metrics = {"trade_score": 0.64}
    assert dl_side_probability(metrics, TradeDirection.CALL) == 0.64
    assert dl_side_probability(metrics, TradeDirection.PUT) == 0.64


def test_veto_blocks_inversion_on_high_conviction_put():
    evaluation = RegimeEvaluation(
        regime=RegimeState.COMPRESSION_TRAP,
        direction_inverted=True,
        trap_boost_score=0.75,
    )
    metrics = {"raw_prob": 0.38}
    result = veto_inversion_on_dl_conviction(evaluation, metrics, TradeDirection.PUT, veto_score=0.60)
    assert result.direction_inverted is False
    assert result.trap_boost_score is None
    assert metrics["dl_inversion_veto"] is True
    assert metrics["direction_inverted"] is False
    assert metrics["dl_side_probability"] == 1.0 - 0.38


def test_veto_ignores_evaluation_without_inversion():
    evaluation = RegimeEvaluation(regime=RegimeState.TREND_EXPANSION, direction_inverted=False)
    metrics = {"raw_prob": 0.90}
    result = veto_inversion_on_dl_conviction(evaluation, metrics, TradeDirection.CALL, veto_score=0.60)
    assert result is evaluation
    assert "dl_inversion_veto" not in metrics


def test_veto_allows_inversion_below_threshold():
    evaluation = RegimeEvaluation(
        regime=RegimeState.CLIMAX_EXHAUSTION,
        direction_inverted=True,
        trap_boost_score=0.76,
    )
    metrics = {"raw_prob": 0.42}
    result = veto_inversion_on_dl_conviction(evaluation, metrics, TradeDirection.PUT, veto_score=0.60)
    assert result is evaluation
    assert result.direction_inverted is True
    assert "dl_inversion_veto" not in metrics
