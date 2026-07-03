from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_micro_position_matrix import (
    classify_micro_zone,
    is_low_consensus,
)
from src.application.services.execution_universal_regime_evaluator import UniversalRegimeEvaluator
from src.application.services.execution_universal_regime_types import (
    MICRO_EXHAUSTION_REGIME_TOKEN,
    MICRO_MIDDLE_UNCERTAINTY_REASON,
    MicroPositionZone,
    RegimeState,
    parse_micro_matrix_cfg,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import base_metrics


def _evaluator(**kwargs) -> UniversalRegimeEvaluator:
    return UniversalRegimeEvaluator({}, **kwargs)


def test_classify_micro_zone_uses_bollinger_percent_b():
    upper = classify_micro_zone(
        {"indicators": {"keltner": 0.5, "bb_pct_b": 0.96}}, keltner_topo=1.05, keltner_fundo=-0.05
    )
    lower = classify_micro_zone(
        {"indicators": {"keltner": 0.5, "bb_pct_b": 0.02}}, keltner_topo=1.05, keltner_fundo=-0.05
    )
    middle = classify_micro_zone(
        {"indicators": {"keltner": 0.5, "bb_pct_b": 0.5}}, keltner_topo=1.05, keltner_fundo=-0.05
    )
    assert upper == MicroPositionZone.UPPER_BOUNDARY
    assert lower == MicroPositionZone.LOWER_BOUNDARY
    assert middle == MicroPositionZone.FAIR_VALUE_MIDDLE


def test_classify_micro_zone_handles_non_numeric_indicators():
    non_numeric = classify_micro_zone(
        {"indicators": {"keltner": "bad", "bb_pct_b": None}},
        keltner_topo=1.05,
        keltner_fundo=-0.05,
    )
    non_finite = classify_micro_zone(
        {"indicators": {"keltner": float("nan"), "bb_pct_b": float("inf")}},
        keltner_topo=1.05,
        keltner_fundo=-0.05,
    )
    assert non_numeric == MicroPositionZone.FAIR_VALUE_MIDDLE
    assert non_finite == MicroPositionZone.FAIR_VALUE_MIDDLE


def test_is_low_consensus_requires_balanced_minority_votes():
    assert is_low_consensus({"call_votes": 4, "put_votes": 2}) is True
    assert is_low_consensus({"call_votes": 3, "put_votes": 1}) is False
    assert is_low_consensus({"call_votes": 6, "put_votes": 2}) is False


def test_parse_micro_matrix_cfg_merges_overrides():
    cfg = parse_micro_matrix_cfg({"keltner_topo_trigger": 1.2, "breakout_momentum_boost": 0.08})
    assert cfg["keltner_topo_trigger"] == 1.2
    assert cfg["breakout_momentum_boost"] == 0.08
    assert cfg["keltner_fundo_trigger"] == -0.05
    assert parse_micro_matrix_cfg(None)["keltner_topo_trigger"] == 1.05


def test_evaluator_respects_configured_keltner_topo_trigger():
    metrics = base_metrics(indicators={"keltner": 0.70})
    evaluator = UniversalRegimeEvaluator({}, micro_cfg={"keltner_topo_trigger": 0.60})
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.micro is not None
    assert evaluation.micro.zone == MicroPositionZone.UPPER_BOUNDARY


def test_micro_matrix_topo_trend_expansion_confirms_call_breakout():
    metrics = base_metrics(
        trade_score=0.82,
        indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10, "keltner": 1.10},
    )
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.TREND_EXPANSION
    assert evaluation.micro.zone == MicroPositionZone.UPPER_BOUNDARY
    result = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.CALL
    assert metrics["micro_position_zone"] == MicroPositionZone.UPPER_BOUNDARY.value
    assert metrics["trade_score"] == 0.87
    assert metrics["micro_breakout_boost"] == 0.05
    assert metrics.get("direction_inverted") is False
    assert metrics["universal_regime"] == RegimeState.TREND_EXPANSION.value


def test_micro_matrix_topo_neutral_inverts_call_to_put_as_exhaustion():
    metrics = base_metrics(indicators={"keltner": 1.10})
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime is None
    assert evaluation.micro.exhaustion_invert is True
    assert evaluation.direction_inverted is True
    inverted = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert inverted == TradeDirection.PUT
    assert metrics["direction_inverted"] is True
    assert metrics["trade_score"] == 0.75
    assert metrics["universal_regime"] == MICRO_EXHAUSTION_REGIME_TOKEN
    assert market_decision_score(metrics) == 0.75


def test_micro_matrix_fundo_trend_expansion_confirms_put_continuation():
    metrics = base_metrics(
        trade_score=0.82,
        call_votes=2,
        put_votes=4,
        indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.45, "cmo": -0.10, "keltner": -0.10},
    )
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.PUT, exec_dir=TradeDirection.PUT)
    assert evaluation.regime == RegimeState.TREND_EXPANSION
    assert evaluation.micro.zone == MicroPositionZone.LOWER_BOUNDARY
    result = evaluator.apply(metrics, evaluation, TradeDirection.PUT, dl_dir=TradeDirection.PUT)
    assert result == TradeDirection.PUT
    assert metrics["trade_score"] == 0.87
    assert metrics["micro_breakout_boost"] == 0.05


def test_micro_matrix_fundo_neutral_inverts_put_to_call_as_exhaustion():
    metrics = base_metrics(indicators={"keltner": -0.10})
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.PUT, exec_dir=TradeDirection.PUT)
    assert evaluation.micro.zone == MicroPositionZone.LOWER_BOUNDARY
    assert evaluation.micro.exhaustion_invert is True
    inverted = evaluator.apply(metrics, evaluation, TradeDirection.PUT, dl_dir=TradeDirection.PUT)
    assert inverted == TradeDirection.CALL
    assert metrics["trade_score"] == 0.75
    assert metrics["universal_regime"] == MICRO_EXHAUSTION_REGIME_TOKEN


def test_micro_matrix_middle_low_consensus_forces_uncertainty_skip():
    metrics = base_metrics()
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime is None
    assert evaluation.micro.middle_uncertainty is True
    result = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.CALL
    assert metrics["micro_position_zone"] == MicroPositionZone.FAIR_VALUE_MIDDLE.value
    assert metrics["trade_score"] == 0.50
    assert metrics["gate_reason"] == MICRO_MIDDLE_UNCERTAINTY_REASON
    assert metrics["regime_skip_cycle"] is True
