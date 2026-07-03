from unittest.mock import patch

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_universal_regime_evaluator import UniversalRegimeEvaluator
from src.application.services.execution_universal_regime_types import RegimeEvaluation, RegimeState
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import base_metrics


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
                "bb_width": 0.008,
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


def test_resolve_execution_direction_compression_trap_bb_expansion_vetoes_inversion():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": base_metrics(
            indicators={
                "vol_ratio": 0.56,
                "adx": 0.17,
                "hurst": 0.48,
                "rsi": 0.63,
                "cmo": 0.10,
                "bb_width": 0.05,
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
    assert direction == TradeDirection.CALL
    assert metrics.get("compression_trap_bb_veto") is True
    assert metrics.get("universal_regime") == "COMPRESSION_TRAP"


def test_resolve_execution_direction_compression_trap_missing_bb_width_vetoes_inversion():
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
    assert direction == TradeDirection.CALL
    assert metrics.get("compression_trap_bb_veto") is True


def test_resolve_execution_direction_realigns_when_regime_apply_leaves_universal_regime_empty():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": base_metrics(deploy_ok=True, val_accuracy=0.58),
    }
    trend_eval = RegimeEvaluation(regime=RegimeState.TREND_EXPANSION)
    with (
        patch.object(UniversalRegimeEvaluator, "evaluate", return_value=trend_eval),
        patch.object(UniversalRegimeEvaluator, "apply", return_value=TradeDirection.CALL),
    ):
        resolved = resolve_execution_direction(
            entry,
            exec_cfg={"regime_evaluator": {"enabled": True}},
        )
    assert resolved is not None
    direction, _ = resolved
    assert direction == TradeDirection.CALL
