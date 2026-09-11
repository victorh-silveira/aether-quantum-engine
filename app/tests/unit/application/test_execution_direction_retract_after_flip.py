"""Finalize: FLIP primeiro; SCALE retract ultima palavra."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.domain.models.trade import TradeDirection


_EXEC_CFG = {
    "scale_vision": {
        "adapt_retract_enabled": True,
        "retraction_require_mili": True,
    }
}


def _base_metrics() -> dict:
    return {
        "raw_prob": 0.62,
        "calibrated_prob": 0.61,
        "conviction": 0.62,
        "trade_score": 0.62,
    }


def _seed_retract_call(_orch, _symbol, _exec_dir, metrics):
    metrics.update(
        {
            "scale_micro_regime": "retraction",
            "scale_micro_side": "CALL",
            "scale_mini_bar_dir": "CALL",
            "scale_mili_dir": "CALL",
            "scale_agree_n": 2,
            "scale_discordance": False,
        }
    )


def _flip_call_to_put(metrics, _tcn_ref, **_kwargs):
    metrics["loss_clf_flip"] = True
    metrics["loss_clf_flip_from"] = "CALL"
    metrics["loss_clf_flip_to"] = "PUT"
    metrics["exec_direction"] = "PUT"
    metrics["resolved_direction"] = "PUT"


def test_finalize_c10_retract_holds_after_flip_to_put():
    metrics = _base_metrics()
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=10)
    with (
        patch(
            "src.application.services.execution_direction_resolver.compute_scale_directions",
            side_effect=_seed_retract_call,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.62),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch("src.application.services.execution_direction_resolver.apply_side_eq_kelly_sizing"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=_flip_call_to_put,
        ),
    ):
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
            exec_cfg=_EXEC_CFG,
        )
    assert out is TradeDirection.CALL
    assert m["exec_direction"] == "CALL"
    assert m["scale_adapted"] is True
    assert m["scale_adapt_reason"] == "retract_holds"
    assert m["scale_adapt_undid_flip"] is True


def test_finalize_flip_preserved_without_retract():
    metrics = _base_metrics()
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=2)

    def _seed_chop(_orch, _symbol, _exec_dir, m):
        m.update(
            {
                "scale_micro_regime": "chop",
                "scale_mini_bar_dir": "CALL",
                "scale_mili_dir": "CALL",
            }
        )

    with (
        patch(
            "src.application.services.execution_direction_resolver.compute_scale_directions",
            side_effect=_seed_chop,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.62),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch("src.application.services.execution_direction_resolver.apply_side_eq_kelly_sizing"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=_flip_call_to_put,
        ),
    ):
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
            exec_cfg=_EXEC_CFG,
        )
    assert out is TradeDirection.PUT
    assert m["scale_adapted"] is False
    assert m.get("loss_clf_flip") is True


def test_finalize_retract_vs_tcn_without_flip():
    metrics = _base_metrics()
    entry = {"metrics": metrics}
    orch = SimpleNamespace(risk_manager=None, stream=None, _active_cycle_id=3)

    def _seed_retract_call_vs_put(_orch, _symbol, _exec_dir, m):
        m.update(
            {
                "scale_micro_regime": "retraction",
                "scale_mini_bar_dir": "CALL",
                "scale_mili_dir": "CALL",
            }
        )

    with (
        patch(
            "src.application.services.execution_direction_resolver.compute_scale_directions",
            side_effect=_seed_retract_call_vs_put,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.PUT, 0.62),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch("src.application.services.execution_direction_resolver.apply_side_eq_kelly_sizing"),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            return_value=False,
        ),
    ):
        out, m = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.PUT,
            0.38,
            0.1,
            meta_applied=False,
            score=0.38,
            symbol="1HZ75V",
            orch=orch,
            exec_cfg=_EXEC_CFG,
        )
    assert out is TradeDirection.CALL
    assert m["scale_adapted"] is True
    assert m["scale_adapt_reason"] == "retract_vs_tcn"
    assert m["scale_adapt_undid_flip"] is False
