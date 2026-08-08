"""Testes do loss-classifier (parte 5): flip bloqueado por seed e scale."""

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
    }
    base.update(overrides)
    return base


def test_bootstrap_seed_high_p_loss_blocks_flip_keeps_tcn():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 4
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "scale_tape_consensus": "PUT",
        "scale_vote_call_n": 0,
        "scale_vote_put_n": 4,
        "kelly_fraction_scale": 1.0,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.95783,
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
    assert metrics["loss_clf_flip_blocked"] == "seed"
    assert metrics["loss_clf_soft"] is True
    assert "loss_clf_flip_block:4" in orch._log_dedupe


def test_scale_consensus_blocks_flip_even_with_auto_learn():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 5
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
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
                "model_version": "loss_123_n40",
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
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "scale_consensus"
