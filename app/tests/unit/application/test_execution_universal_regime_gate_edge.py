from math import inf, nan
from unittest.mock import MagicMock, patch

from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics
from src.application.services.execution_universal_regime_evaluator import UniversalRegimeEvaluator
from src.application.services.execution_universal_regime_gate import (
    apply_universal_regime_penalty_to_metrics,
    evaluate_universal_loss_scenarios,
    log_regime_audit,
    map_volatility_regime,
)
from src.application.services.execution_universal_regime_types import (
    RegimeState,
    UniversalLossEvaluation,
)
from src.application.services.orchestrator.execution_collect_gather import (
    _dl_direction_from_metrics,
    gather_cluster_candidates,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import base_metrics


def test_evaluate_universal_loss_scenarios_no_match_neutral():
    metrics = base_metrics(indicators={"vol_ratio": 0.90, "adx": 0.21, "hurst": 0.52, "rsi": 0.50, "cmo": 0.05})
    evaluation = evaluate_universal_loss_scenarios(
        metrics,
        dl_dir=TradeDirection.CALL,
        exec_dir=TradeDirection.PUT,
    )
    assert evaluation.scenario is None


def test_evaluate_universal_loss_scenarios_missing_direction():
    metrics = base_metrics(dl_direction=None, exec_direction=None)
    evaluation = evaluate_universal_loss_scenarios(metrics)
    assert evaluation.scenario is None


def test_evaluate_universal_loss_scenarios_nan_inf_indicators():
    metrics = base_metrics(indicators={"vol_ratio": nan, "adx": inf, "rsi": "bad", "cmo": None, "hurst": nan})
    evaluation = evaluate_universal_loss_scenarios(
        metrics,
        dl_dir=TradeDirection.CALL,
        exec_dir=TradeDirection.CALL,
    )
    assert evaluation.scenario in (None, RegimeState.ENTROPIC_NOISE.value)


def test_apply_universal_regime_penalty_noop():
    metrics = base_metrics()
    factor = apply_universal_regime_penalty_to_metrics(
        metrics,
        UniversalLossEvaluation(scenario=None, score_factor=1.0),
    )
    assert factor == 1.0
    assert "universal_regime_score_factor" not in metrics


def test_apply_quality_penalty_skips_when_regime_already_applied():
    metrics = base_metrics(universal_regime="TREND_EXPANSION", universal_regime_score_factor=1.0)
    apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.56,
        min_val=0.60,
        min_edge=0.04,
    )
    assert metrics["universal_regime"] == "TREND_EXPANSION"


def test_apply_quality_penalty_to_metrics_entropic_noise_legacy_path():
    metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        val_accuracy=0.70,
        edge=0.10,
        direction_margin=0.08,
        indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.56,
        min_val=0.60,
        min_edge=0.04,
    )
    assert metrics.get("universal_regime") == RegimeState.ENTROPIC_NOISE.value
    assert metrics.get("gate_penalty") == "noise"


def test_evaluate_universal_loss_scenarios_invalid_dl_direction_in_metrics():
    metrics = base_metrics(dl_direction="INVALID", exec_direction="INVALID")
    evaluation = evaluate_universal_loss_scenarios(metrics)
    assert evaluation.scenario is None


def test_apply_quality_penalty_invalid_direction_strings():
    metrics = base_metrics(
        dl_direction="NOT_A_DIRECTION",
        exec_direction="ALSO_BAD",
        val_accuracy=0.70,
        edge=0.10,
        direction_margin=0.08,
    )
    penalty = apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.56,
        min_val=0.60,
        min_edge=0.04,
    )
    assert penalty == 0.0


def test_evaluator_prefers_macro_indicators_for_regime():
    metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        indicators={"hurst": 0.55, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
        macro_indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    evaluator = UniversalRegimeEvaluator({}, recovery_active=True, mandatory_min_signal=0.56)
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime is not None


def test_log_regime_audit_emits_when_regime_present():
    logger = MagicMock()
    metrics = base_metrics(universal_regime="CLIMAX_EXHAUSTION", direction_inverted=True)
    log_regime_audit(logger, "C0042", "RDBULL", TradeDirection.CALL, TradeDirection.PUT, metrics)
    logger.info.assert_called_once()
    args = logger.info.call_args[0]
    message = args[0] % args[1:]
    assert "REGIME MACRO" in message
    assert "CLIMAX_EXHAUSTION" in message
    assert "M1 PUT" in message


def test_log_regime_audit_skips_without_regime():
    logger = MagicMock()
    log_regime_audit(logger, "C0001", "RDBEAR", TradeDirection.CALL, TradeDirection.CALL, base_metrics())
    logger.info.assert_not_called()


def test_evaluator_disabled_returns_neutral():
    evaluator = UniversalRegimeEvaluator({"enabled": False})
    metrics = base_metrics(indicators={"adx": 0.30, "rsi": 0.75, "cmo": 0.50})
    evaluation = evaluator.evaluate(metrics, dl_dir=TradeDirection.CALL, exec_dir=TradeDirection.CALL)
    assert evaluation.regime is None


def test_map_volatility_regime_handles_non_dict_indicators():
    assert map_volatility_regime({"indicators": "invalid"}) == "neutral"


def test_gather_dl_direction_helper_fallback_paths():
    assert _dl_direction_from_metrics({}, TradeDirection.PUT) == TradeDirection.PUT
    assert _dl_direction_from_metrics({"dl_direction": "INVALID"}, TradeDirection.CALL) == TradeDirection.CALL


def test_gather_cluster_candidates_uses_dl_direction_fallback():
    exec_mgr = MagicMock()
    exec_mgr.orch.config = {
        "orchestrator": {
            "execution": {
                "regime_evaluator": {"enabled": True},
                "mandatory_trade_each_cycle": True,
            }
        },
        "deep_learning": {"calibration": {}},
    }
    exec_mgr._trade_symbols.return_value = ["RDBEAR"]
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": base_metrics(
            dl_direction="INVALID",
            deploy_ok=True,
            execute=True,
            val_accuracy=0.60,
            edge=0.08,
            indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10},
        ),
    }
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.cluster_entry_eligible",
        return_value=True,
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            {"RDBEAR": entry},
            recovery_active=False,
            recovery_cfg={},
            cid="C0010",
            min_signal=0.56,
            min_val=0.50,
        )
    assert len(candidates) == 1
    exec_mgr.logger.info.assert_called()


def test_gather_cluster_candidates_skips_entropic_regime_skip():
    exec_mgr = MagicMock()
    exec_mgr.orch.config = {
        "orchestrator": {
            "execution": {
                "regime_evaluator": {"enabled": True},
            }
        },
        "deep_learning": {"calibration": {}},
    }
    exec_mgr._trade_symbols.return_value = ["RDBEAR"]
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": base_metrics(
            call_votes=3,
            put_votes=3,
            deploy_ok=True,
            execute=True,
            val_accuracy=0.60,
            edge=0.08,
            indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
        ),
    }
    decisions = {"RDBEAR": entry}
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.cluster_entry_eligible",
        return_value=True,
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            decisions,
            recovery_active=False,
            recovery_cfg={},
            cid="C0007",
            min_signal=0.56,
            min_val=0.50,
        )
    assert candidates == []
