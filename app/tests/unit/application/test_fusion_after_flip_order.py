"""Fusao nao pode desfazer FLIP do loss-clf (ordem fusion -> flip TCN)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.application.services.loss_classifier_gate_support import resolve_tcn_ref
from src.domain.models.trade import TradeDirection


def test_resolve_tcn_ref_falls_back_to_dl_direction():
    assert resolve_tcn_ref({"dl_direction": "PUT"}, TradeDirection.CALL) == TradeDirection.PUT
    assert resolve_tcn_ref({"tcn_direction": "CALL"}, TradeDirection.PUT) == TradeDirection.CALL
    assert resolve_tcn_ref({}, TradeDirection.CALL) == TradeDirection.CALL


def test_finalize_loss_flip_survives_fusion_ev_call():
    metrics = {
        "raw_prob": 0.45,
        "calibrated_prob": 0.57,
        "conviction": 0.57,
        "trade_score": 0.57,
        "dl_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "execution_candidate_ready": True,
    }
    entry = {"metrics": metrics}
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"scale_vision": {}}}},
        risk_manager=None,
        stream=None,
        state=SimpleNamespace(balance=9650.0),
        _active_cycle_id=1,
        _log_dedupe={},
    )
    order: list[str] = []

    def _fusion(m, d, **_k):
        order.append("fusion")
        m["fusion_applied"] = True
        m["fusion_reason"] = "ev_call"
        m["fusion_switched"] = False
        m["exec_direction"] = TradeDirection.CALL.name
        m["resolved_direction"] = TradeDirection.CALL.name
        return TradeDirection.CALL

    def _loss_gate(m, ref, **_k):
        order.append("loss")
        assert ref == TradeDirection.CALL
        m["loss_clf_flip"] = True
        m["loss_clf_flip_ref"] = TradeDirection.CALL.name
        m["loss_clf_flip_reason"] = "p_ovr"
        m["loss_clf_auto_learn"] = False
        m["loss_clf_p_loss"] = 0.95978
        m["exec_direction"] = TradeDirection.PUT.name
        m["resolved_direction"] = TradeDirection.PUT.name
        return False

    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.57),
        ),
        patch("src.application.services.execution_direction_resolver.attach_live_signal_metrics"),
        patch("src.application.services.execution_direction_resolver.apply_live_calib_drift_soft"),
        patch("src.application.services.execution_direction_resolver.ensure_direction_margin"),
        patch("src.application.services.execution_direction_resolver.compute_scale_directions"),
        patch(
            "src.application.services.execution_direction_resolver.parse_direction_fusion_config",
            return_value={"fusion_enabled": True, "fusion_replace_adapt_flip": True},
        ),
        patch("src.application.services.execution_direction_resolver.apply_scale_kelly_side_sync"),
        patch("src.application.services.execution_direction_resolver.apply_side_eq_kelly_sizing"),
        patch("src.application.services.execution_direction_resolver.apply_scale_kelly_sizing"),
        patch("src.application.services.execution_direction_resolver.sync_direction_margin"),
        patch(
            "src.application.services.execution_direction_resolver.apply_signal_skip_gates",
            side_effect=lambda m, d, **k: None,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_direction_fusion",
            side_effect=_fusion,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            side_effect=_loss_gate,
        ),
        patch("src.application.services.execution_direction_resolver.apply_regime_chop_pause"),
        patch("src.application.services.execution_direction_resolver.apply_negative_cal_edge_pause"),
        patch(
            "src.application.services.execution_direction_resolver.apply_invert_exec_side",
            side_effect=lambda m, d, **k: d,
        ),
    ):
        exec_dir, out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.57,
            0.0,
            meta_applied=False,
            score=0.57,
            symbol="R_10",
            orch=orch,
        )

    assert order == ["fusion", "loss"]
    assert exec_dir == TradeDirection.PUT
    assert out["exec_direction"] == "PUT"
    assert out["loss_clf_flip"] is True
    assert out["tcn_direction"] == "CALL"
    assert out.get("fusion_reason") == "ev_call"
