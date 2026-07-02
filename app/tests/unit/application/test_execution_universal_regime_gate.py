from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_universal_regime_evaluator import UniversalRegimeEvaluator
from src.application.services.execution_universal_regime_gate import (
    UniversalLossEvaluation,
    apply_compression_trap_boost,
    apply_regime_direction_boost,
    apply_universal_regime_penalty_to_metrics,
    apply_universal_regime_resolution,
    evaluate_universal_loss_scenarios,
    invert_trade_direction,
    map_volatility_regime,
    map_volatility_regime_to_metrics,
)
from src.application.services.execution_universal_regime_types import (
    CLIMAX_EXHAUSTION_SCORE,
    COMPRESSION_TRAP_SCORE,
    RegimeState,
    parse_regime_evaluator_cfg,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import base_metrics


def _evaluator(**kwargs) -> UniversalRegimeEvaluator:
    return UniversalRegimeEvaluator({}, **kwargs)


def test_apply_regime_direction_boost_call_branch():
    metrics = base_metrics()
    apply_regime_direction_boost(metrics, TradeDirection.CALL, COMPRESSION_TRAP_SCORE, "COMPRESSION_TRAP")
    assert metrics["direction_call_score"] == COMPRESSION_TRAP_SCORE
    assert metrics["direction_put_score"] == 0.25


def test_map_volatility_regime_uses_macro_indicators():
    metrics = {"macro_indicators": {"vol_ratio": 0.70}, "indicators": {"vol_ratio": 1.20}}
    assert map_volatility_regime(metrics) == "compression"


def test_entropic_noise_uses_dynamic_threshold_pivot():
    metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        calibrated_prob=None,
        raw_prob=0.44,
        dynamic_call_threshold=0.53,
        dynamic_put_threshold=0.47,
        indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    evaluator = _evaluator(recovery_active=True, mandatory_min_signal=0.56)
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.ENTROPIC_NOISE
    assert evaluation.regime_skip_cycle is True
    result = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.CALL
    assert metrics.get("regime_skip_cycle") is True


def test_evaluate_universal_loss_scenarios_legacy_wrapper():
    metrics = base_metrics(indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10})

    evaluation = evaluate_universal_loss_scenarios(
        metrics,
        dl_dir=TradeDirection.CALL,
        exec_dir=TradeDirection.CALL,
    )
    assert evaluation.scenario == "TREND_EXPANSION"


def test_apply_universal_regime_penalty_and_resolution_legacy():
    metrics = base_metrics()
    penalty_eval = UniversalLossEvaluation(scenario="ENTROPIC_NOISE", score_factor=0.50, gate_penalty="noise")
    factor = apply_universal_regime_penalty_to_metrics(metrics, penalty_eval)
    assert factor == 0.50
    invert_eval = UniversalLossEvaluation(
        scenario="COMPRESSION_TRAP",
        score_factor=1.0,
        invert_direction=True,
        trap_boost_score=COMPRESSION_TRAP_SCORE,
    )
    resolved = apply_universal_regime_resolution(metrics, invert_eval, TradeDirection.CALL)
    assert resolved == TradeDirection.PUT


def test_parse_regime_evaluator_cfg_merges_float_keys():
    cfg = parse_regime_evaluator_cfg({"trend_adx_min": 0.25, "enabled": False})
    assert cfg["trend_adx_min"] == 0.25
    assert cfg["enabled"] is False


def test_entropic_noise_without_probability_keeps_exec_dir_on_skip():
    metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    metrics.pop("calibrated_prob")
    metrics.pop("raw_prob")
    evaluator = _evaluator(recovery_active=True, mandatory_min_signal=0.56)
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.PUT)
    assert evaluation.regime_skip_cycle is True
    result = evaluator.apply(metrics, evaluation, TradeDirection.PUT, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.PUT
    assert metrics.get("regime_skip_cycle") is True


def test_apply_universal_regime_resolution_no_scenario_returns_unchanged():
    metrics = base_metrics()
    evaluation = UniversalLossEvaluation(scenario=None, score_factor=1.0)
    assert apply_universal_regime_resolution(metrics, evaluation, TradeDirection.CALL) == TradeDirection.CALL


def test_apply_universal_regime_resolution_invalid_scenario_returns_unchanged():
    metrics = base_metrics()
    evaluation = UniversalLossEvaluation(scenario="invalid_legacy", score_factor=1.0, invert_direction=True)
    assert apply_universal_regime_resolution(metrics, evaluation, TradeDirection.CALL) == TradeDirection.CALL


def test_map_volatility_regime_treats_nan_as_neutral():
    assert map_volatility_regime({"indicators": {"vol_ratio": float("nan")}}) == "neutral"


def test_apply_compression_trap_boost_wrapper():
    metrics = base_metrics()
    apply_compression_trap_boost(metrics, TradeDirection.CALL, COMPRESSION_TRAP_SCORE)
    assert metrics["universal_regime"] == "COMPRESSION_TRAP"


def test_apply_universal_regime_penalty_applies_mandatory_floor():
    metrics = base_metrics(trade_score=0.82, conviction=0.82, resolved_conviction=0.82)
    evaluation = UniversalLossEvaluation(
        scenario="ENTROPIC_NOISE",
        score_factor=0.50,
        mandatory_conviction_floor=0.56,
    )
    apply_universal_regime_penalty_to_metrics(metrics, evaluation)
    assert metrics["trade_score"] == 0.56


def test_apply_universal_regime_resolution_uses_exec_dir_when_dl_invalid():
    metrics = base_metrics(
        dl_direction="BAD",
        indicators={"adx": 0.17, "hurst": 0.48, "vol_ratio": 0.56, "rsi": 0.62, "cmo": 0.05},
    )
    evaluation = UniversalLossEvaluation(
        scenario="COMPRESSION_TRAP",
        score_factor=1.0,
        invert_direction=True,
        trap_boost_score=COMPRESSION_TRAP_SCORE,
    )
    resolved = apply_universal_regime_resolution(metrics, evaluation, TradeDirection.CALL)
    assert resolved == TradeDirection.PUT


def test_map_volatility_regime_compression_expansion_neutral():
    assert map_volatility_regime({"indicators": {"vol_ratio": 0.70}}) == "compression"
    assert map_volatility_regime({"indicators": {"vol_ratio": 1.20}}) == "expansion"
    assert map_volatility_regime({"indicators": {"vol_ratio": 1.0}}) == "neutral"
    metrics = {"indicators": {"vol_ratio": 0.70}}
    assert map_volatility_regime_to_metrics(metrics) == "compression"
    assert metrics["vol_regime"] == "compression"


def test_trend_expansion_preserves_direction():
    metrics = base_metrics(indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10})
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.TREND_EXPANSION
    assert evaluation.direction_inverted is False
    result = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.CALL
    assert metrics.get("direction_inverted") is False


def test_compression_trap_inverts_call_to_put_with_boosted_score():
    metrics = base_metrics(
        indicators={
            "adx": 0.17,
            "hurst": 0.48,
            "vol_ratio": 0.56,
            "rsi": 0.63,
            "cmo": 0.05,
        }
    )
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.COMPRESSION_TRAP
    assert evaluation.direction_inverted is True
    inverted = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert inverted == TradeDirection.PUT
    assert metrics["compression_trap_inverted"] is True
    assert metrics["trade_score"] == COMPRESSION_TRAP_SCORE
    assert market_decision_score(metrics) == COMPRESSION_TRAP_SCORE


def test_compression_trap_no_invert_without_band_stretch():
    metrics = base_metrics(
        indicators={
            "adx": 0.17,
            "hurst": 0.48,
            "vol_ratio": 0.56,
            "rsi": 0.50,
            "cmo": 0.05,
        }
    )
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.COMPRESSION_TRAP
    assert evaluation.direction_inverted is False
    result = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert result == TradeDirection.CALL
    assert metrics.get("market_decision_score_override") is None


def test_climax_exhaustion_inverts_call_to_put():
    metrics = base_metrics(indicators={"adx": 0.30, "rsi": 0.72, "cmo": 0.54, "hurst": 0.55, "vol_ratio": 1.05})
    evaluator = _evaluator()
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.CLIMAX_EXHAUSTION
    inverted = evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert inverted == TradeDirection.PUT
    assert metrics["trade_score"] == CLIMAX_EXHAUSTION_SCORE


def test_entropic_noise_skips_cycle_outside_recovery():
    metrics = base_metrics(call_votes=3, put_votes=3, indicators={"hurst": 0.50, "adx": 0.18})
    evaluator = _evaluator(recovery_active=False, continuous_mode=False)
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.ENTROPIC_NOISE
    assert evaluation.regime_skip_cycle is True
    evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert metrics.get("regime_skip_cycle") is True


def test_entropic_noise_tied_votes_forces_skip_in_recovery():
    metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        raw_prob=0.68,
        calibrated_prob=0.68,
        indicators={"hurst": 0.40, "adx": 0.18},
    )
    evaluator = _evaluator(recovery_active=True, continuous_mode=True, mandatory_min_signal=0.56)
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime == RegimeState.ENTROPIC_NOISE
    assert evaluation.regime_skip_cycle is True
    evaluator.apply(metrics, evaluation, TradeDirection.CALL, dl_dir=TradeDirection.CALL)
    assert metrics.get("regime_skip_cycle") is True


def test_apply_regime_direction_boost_put_branch():
    metrics = base_metrics()
    apply_regime_direction_boost(metrics, TradeDirection.PUT, COMPRESSION_TRAP_SCORE, "COMPRESSION_TRAP")
    assert metrics["direction_put_score"] == COMPRESSION_TRAP_SCORE
    assert metrics["direction_call_score"] == 0.25


def test_invert_trade_direction():
    assert invert_trade_direction(TradeDirection.PUT) == TradeDirection.CALL


def test_resolve_execution_direction_applies_compression_trap_inversion():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": base_metrics(
            indicators={
                "vol_ratio": 0.56,
                "adx": 0.17,
                "hurst": 0.48,
                "rsi": 0.63,
                "cmo": 0.10,
            },
            deploy_ok=True,
            val_accuracy=0.58,
        ),
    }
    resolved = resolve_execution_direction(
        entry,
        exec_cfg={"regime_evaluator": {"enabled": True}},
    )
    assert resolved is not None
    direction, metrics = resolved
    assert direction == TradeDirection.PUT
    assert metrics.get("universal_regime") == "COMPRESSION_TRAP"
