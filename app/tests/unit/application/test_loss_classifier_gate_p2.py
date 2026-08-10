"""Testes do loss-classifier (parte 2): soft faixa media e flip >= floor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def _soft_cfg(**overrides):
    base = {
        "veto_mode": "soft",
        "veto_p_loss_floor": 0.65,
        "hard_p_loss_floor": 0.90,
        "hard_blocks_pending_waive": True,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.20,
        "soft_p_loss_high": 0.85,
        "soft_max_stake_pct_high": 0.0025,
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_scale_discord": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_cal_discord_margin": 0.03,
        "flip_require_pos_edge": False,
        "flip_min_edge_execute": 0.04,
    }
    base.update(overrides)
    return base


def test_apply_loss_classifier_veto_and_ok_paths():
    metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "scale_adapted": True,
        "scale_micro_regime": "chop",
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "kelly_fraction_scale": 1.0,
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = "bad"
    orch._log_dedupe = {}
    veto_payload = {
        "p_loss": 0.75,
        "veto": True,
        "auto_learn_applied": True,
        "model_version": "auto1",
        "n_train": 40,
        "veto_ready": True,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=veto_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics.get("gate_reason") is None
    assert metrics["loss_clf_soft"] is True
    assert metrics["kelly_fraction_scale"] < 1.0
    soft_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "scale_micro_regime": "chop",
    }
    soft_payload = dict(veto_payload)
    soft_payload["p_loss"] = 0.65
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=soft_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(soft_metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert soft_metrics.get("gate_reason") is None
    assert soft_metrics.get("execution_candidate_ready") is not False
    assert soft_metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert soft_metrics["loss_clf_soft"] is True
    assert soft_metrics["loss_clf_soft_max_stake_pct"] == pytest.approx(0.0025)
    pending_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
    }
    orch.risk_manager.pending_loss_total = MagicMock(return_value=38.0)
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=dict(veto_payload, p_loss=0.75),
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(pending_metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert pending_metrics["loss_clf_soft"] is True
    assert pending_metrics.get("loss_clf_soft_waived_pending") is True
    assert pending_metrics["kelly_fraction_scale"] == pytest.approx(1.0)
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    high_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
        "execution_candidate_ready": True,
    }
    high_payload = dict(veto_payload)
    high_payload["p_loss"] = 0.92
    high_payload["veto_ready"] = True
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=high_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(high_metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert high_metrics["execution_candidate_ready"] is True
    assert high_metrics["exec_direction"] == "PUT"
    assert high_metrics["resolved_direction"] == "PUT"
    assert high_metrics["loss_clf_flip"] is True
    assert high_metrics["loss_clf_flip_ref"] == "CALL"
    assert high_metrics["loss_clf_veto_mode"] == "flip"
    assert high_metrics.get("gate_reason") is None
    metrics2 = dict(metrics)
    metrics2.pop("gate_reason", None)
    metrics2.pop("signal_skip_reason", None)
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.2,
            "veto": False,
            "auto_learn_applied": False,
            "model_version": "m",
            "n_train": 5,
            "veto_ready": False,
        },
    ):
        assert apply_loss_classifier_gate(metrics2, TradeDirection.PUT, orch=orch, symbol="R_10") is False


def test_flip_call_to_put_and_put_to_call_with_pending():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 2
    orch.risk_manager.pending_loss_total = MagicMock(return_value=50.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    metrics = {
        "kelly_fraction_scale": 1.0,
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.9421,
                "veto": True,
                "auto_learn_applied": True,
                "model_version": "loss_live_n40",
                "n_train": 40,
                "veto_ready": True,
                "bootstrap": False,
            },
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics["exec_direction"] == "CALL"
    assert metrics["loss_clf_flip"] is True
    assert metrics["loss_clf_flip_ref"] == "PUT"
    assert metrics.get("loss_clf_soft_waived_pending") is False
    assert metrics["execution_candidate_ready"] is True


def test_flip_anchors_tcn_not_scale_adapted_side():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 9
    metrics = {
        "kelly_fraction_scale": 1.0,
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "CALL",
        "scale_adapted": True,
    }
    captured: dict[str, object] = {}

    def _capture(_config, payload):
        captured["direction"] = payload["direction"]
        return {
            "p_loss": 0.91,
            "veto": True,
            "auto_learn_applied": True,
            "model_version": "loss_live_n40",
            "n_train": 40,
            "veto_ready": True,
            "bootstrap": False,
        }

    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            side_effect=_capture,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
        patch(
            "src.application.services.loss_classifier_gate.build_loss_feature_vector",
            return_value=[0.0] * 24,
        ) as build_mock,
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert build_mock.call_args.args[1] == TradeDirection.CALL
    assert captured["direction"] == "CALL"
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip"] is True
    assert metrics["loss_clf_flip_ref"] == "CALL"
    assert "loss_clf_flip:9" in orch._log_dedupe


def test_skip_before_loss_and_force_and_disabled():
    assert apply_loss_classifier_gate({"gate_reason": "cal_margin"}, TradeDirection.CALL, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, force=True, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=None) is False
    with patch("src.application.services.loss_classifier_gate.loss_classifier_enabled", return_value=False):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=MagicMock()) is False
