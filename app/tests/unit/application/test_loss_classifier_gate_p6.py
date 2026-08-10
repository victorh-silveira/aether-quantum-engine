"""Testes do loss-classifier (parte 6): edge pos-FLIP e override auto_learn."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
    }
    base.update(overrides)
    return base


def _orch(cycle_id: int) -> MagicMock:
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = cycle_id
    return orch


def test_auto_learn_cal_override_then_neg_edge_reverts():
    orch = _orch(10)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.58,
        "scale_tape_consensus": "PUT",
        "scale_vote_call_n": 0,
        "scale_vote_put_n": 4,
        "kelly_fraction_scale": 1.0,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.95,
                "veto": True,
                "auto_learn_applied": True,
                "model_version": "loss_live_n48",
                "n_train": 48,
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
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "neg_edge"
    assert metrics.get("loss_clf_flip_cal_overrides_scale") is True
    assert metrics["loss_clf_soft"] is True


def test_auto_learn_cal_override_with_pos_edge_flips():
    orch = _orch(11)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.62,
        "scale_tape_consensus": "PUT",
        "scale_vote_call_n": 0,
        "scale_vote_put_n": 4,
        "kelly_fraction_scale": 1.0,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.95,
                "veto": True,
                "auto_learn_applied": True,
                "model_version": "loss_live_n48",
                "n_train": 48,
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
    assert metrics.get("loss_clf_flip") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics.get("loss_clf_flip_reason") == "cal_ovr"


def test_post_flip_neg_edge_blocks_without_scale():
    orch = _orch(12)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.55,
        "scale_tape_consensus": "CALL",
        "scale_vote_call_n": 3,
        "scale_vote_put_n": 1,
        "kelly_fraction_scale": 1.0,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.95,
                "veto": True,
                "auto_learn_applied": False,
                "model_version": "loss_bootstrap_synth",
                "n_train": 64,
                "veto_ready": True,
                "bootstrap": True,
            },
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "neg_edge"
    assert metrics["loss_clf_soft"] is True


def test_collapsed_p_loss_disables_veto_and_logs_degen():
    orch = _orch(8)
    metrics = {"execution_candidate_ready": True, "exec_direction": "CALL"}
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.5,
                "veto": False,
                "auto_learn_applied": True,
                "model_version": "loss_live",
                "n_train": 8,
                "veto_ready": True,
                "bootstrap": False,
            },
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(flip_require_auto_learn=False, veto_p_loss_floor=0.99, hard_p_loss_floor=0.99),
        ),
        patch.object(orch, "logger", MagicMock()),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_collapsed") is True
    assert metrics.get("loss_clf_veto_ready") is False
