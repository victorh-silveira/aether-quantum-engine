"""Finalize: TCN + FLIP; sem SCALE adapt nem skips de vela."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.domain.models.trade import TradeDirection


def _base_metrics() -> dict:
    return {
        "raw_prob": 0.62,
        "calibrated_prob": 0.61,
        "conviction": 0.62,
        "trade_score": 0.62,
        "cal_side_edge": 0.12,
    }


def _flip_call_to_put(metrics, _tcn_ref, **_kwargs):
    metrics["loss_clf_flip"] = True
    metrics["loss_clf_flip_from"] = "CALL"
    metrics["loss_clf_flip_to"] = "PUT"
    metrics["exec_direction"] = "PUT"
    metrics["resolved_direction"] = "PUT"


def _stack_patches(stack: ExitStack, *, flip_side_effect=None) -> None:
    flip = flip_side_effect if flip_side_effect is not None else (lambda *_a, **_k: False)
    stack.enter_context(
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.62),
        )
    )
    stack.enter_context(patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"))
    stack.enter_context(patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"))
    stack.enter_context(patch("src.application.services.execution_direction_resolver.ensure_direction_margin"))
    stack.enter_context(patch("src.application.services.execution_direction_resolver.sync_direction_margin"))
    stack.enter_context(patch("src.application.services.execution_direction_resolver.apply_side_eq_kelly_sizing"))
    stack.enter_context(
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=flip,
        )
    )
    stack.enter_context(
        patch(
            "src.application.services.execution_direction_resolver.should_skip_neg_edge",
            return_value=False,
        )
    )


def test_finalize_keeps_flip_without_scale_adapt():
    metrics = _base_metrics()
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=10)
    with ExitStack() as stack:
        _stack_patches(stack, flip_side_effect=_flip_call_to_put)
        out, m = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="1HZ75V",
            orch=orch,
            exec_cfg={},
        )
    assert out is TradeDirection.PUT
    assert m["exec_direction"] == "PUT"
    assert m.get("scale_adapt_applied") is False
    assert m.get("loss_clf_flip") is True


def test_finalize_keeps_tcn_when_no_flip():
    metrics = _base_metrics()
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=2)
    with ExitStack() as stack:
        _stack_patches(stack)
        out, m = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="1HZ75V",
            orch=orch,
            exec_cfg={},
        )
    assert out is TradeDirection.CALL
    assert m["exec_direction"] == "CALL"
    assert m.get("scale_adapt_applied") is False


def test_finalize_neg_edge_skips():
    metrics = _base_metrics()
    metrics["cal_side_edge"] = -0.01
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=3)
    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.62),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_skip_neg_edge",
            return_value=True,
        ),
    ):
        out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="1HZ75V",
            orch=orch,
            exec_cfg={"skip_neg_edge": True, "senior_confluence_flip": False},
        )
    assert out is None


def test_finalize_ignores_candle_discord_metrics():
    metrics = _base_metrics()
    metrics.update(
        {
            "closed_micro_candle_stamped": True,
            "closed_micro_candle_dir": "PUT",
            "scale_adapt_reason": "candle_vs_tcn",
        }
    )
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=4)
    with ExitStack() as stack:
        _stack_patches(stack)
        out, m = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.62,
            0.1,
            meta_applied=False,
            score=0.62,
            symbol="1HZ75V",
            orch=orch,
            exec_cfg={},
        )
    assert out is TradeDirection.CALL
    assert m["exec_direction"] == "CALL"
