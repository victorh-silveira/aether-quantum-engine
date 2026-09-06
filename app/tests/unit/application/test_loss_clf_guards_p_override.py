"""Override de guards FLIP quando p_loss >= flip_waive_guards_above_p_loss."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.application.services.execution_micro_protect import apply_micro_discord_hard_skip
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.loss_classifier_gate_support import (
    apply_flip_guards_p_override,
    flip_guards_p_override,
)
from src.domain.models.trade import TradeDirection


def test_flip_guards_p_override_threshold():
    cfg = {"flip_waive_guards_above_p_loss": 0.85}
    assert flip_guards_p_override(0.857, cfg) is True
    assert flip_guards_p_override(0.80, cfg) is False
    assert flip_guards_p_override(0.85, {"flip_waive_guards_above_p_loss": "x"}) is False
    assert flip_guards_p_override(0.99, {"flip_waive_guards_above_p_loss": 0.0}) is False
    assert flip_guards_p_override(0.99, {}) is False


def test_resolve_rejects_bad_guards_override():
    import pytest

    from src.infrastructure.inference.loss_classifier_client import resolve_loss_classifier_config

    with pytest.raises(ValueError, match="flip_waive_guards_above_p_loss"):
        resolve_loss_classifier_config({"flip_waive_guards_above_p_loss": 0.10})


def test_apply_flip_guards_clears_blocks_and_floor():
    metrics: dict = {}
    seed, scale, pos, candle, floor = apply_flip_guards_p_override(
        metrics,
        p_loss=0.96,
        cfg={"flip_waive_guards_above_p_loss": 0.85},
        seed_block=True,
        scale_block=True,
        pos_edge_block=True,
        seed_candle_block=True,
        flip_floor=0.90,
    )
    assert (seed, scale, pos, candle) == (False, False, False, False)
    assert floor == 0.85
    assert metrics["loss_clf_flip_guards_p_override"] is True
    assert metrics["loss_clf_flip_tcn_edge_p_override"] is True


def test_gate_c1_like_flips_put_to_call_with_aligned_candle():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 9000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 1
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.35,
        "raw_prob": 0.37,
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "scale_vote_call_n": 0,
        "scale_vote_put_n": 3,
        "kelly_fraction_scale": 1.0,
    }
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.95783,
            "model_version": "seed-c1",
            "n_train": 64,
            "auto_learn_applied": False,
            "veto_ready": True,
            "collapsed": False,
        },
    ):
        apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="1HZ75V")
    assert metrics.get("loss_clf_flip") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics.get("loss_clf_flip_guards_p_override") is True


def test_gate_c3_like_flips_at_p_loss_0857():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 9000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 3
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.355,
        "raw_prob": 0.40,
        "closed_micro_candle_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "kelly_fraction_scale": 1.0,
    }
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.85686,
            "model_version": "seed-c3",
            "n_train": 64,
            "auto_learn_applied": False,
            "veto_ready": True,
            "collapsed": False,
        },
    ):
        apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="1HZ75V")
    assert metrics.get("loss_clf_flip") is True
    assert metrics["exec_direction"] == "CALL"


def test_gate_below_override_keeps_seed_block():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 9000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 9
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.35,
        "raw_prob": 0.35,
        "closed_micro_candle_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "kelly_fraction_scale": 1.0,
    }
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.80,
            "model_version": "seed-low",
            "n_train": 64,
            "auto_learn_applied": False,
            "veto_ready": True,
            "collapsed": False,
        },
    ):
        apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="1HZ75V")
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"


def test_micro_skips_after_loss_clf_flip():
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 5.0,
        "loss_clf_flip": True,
        "calibrated_prob": 0.35,
    }
    assert apply_micro_discord_hard_skip(metrics) is False
    assert metrics.get("gate_reason") != "micro_discord"


def test_neg_edge_soft_after_flip_even_if_candle_disagrees():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.35,
        "kelly_fraction_scale": 1.0,
        "loss_clf_flip": True,
        "closed_micro_candle_dir": "PUT",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.015},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": True,
                    "neg_edge_soft_min_edge": -1.0,
                    "neg_edge_bootstrap_soft_kelly_mult": 0.25,
                    "neg_edge_deep_edge_floor": -0.12,
                }
            }
        },
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is False
    assert metrics["neg_edge_soft"] is True
    assert metrics["gate_verdict"] == "SOFT_SIZE"
