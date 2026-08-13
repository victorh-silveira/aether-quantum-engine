"""Testes seed candle block, seed edge min e neg_edge bootstrap profundo."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.loss_classifier_flip import (
    post_flip_edge_ok,
    seed_candle_blocks_flip,
)
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def test_seed_candle_blocks_flip_when_candle_agrees_tcn():
    metrics = {"closed_micro_candle_dir": "CALL"}
    response = {"auto_learn_applied": False}
    cfg = {"flip_seed_block_against_closed_candle": True}
    assert seed_candle_blocks_flip(metrics, response, TradeDirection.CALL, cfg=cfg) is True
    assert metrics.get("loss_clf_flip_block_seed_candle") is True
    live = {"closed_micro_candle_dir": "CALL"}
    assert (
        seed_candle_blocks_flip(
            live,
            {"auto_learn_applied": True},
            TradeDirection.CALL,
            cfg=cfg,
        )
        is False
    )


def test_post_flip_edge_ok_uses_seed_waive_min():
    metrics = {
        "calibrated_prob": 0.56,
        "closed_micro_candle_dir": "CALL",
        "loss_clf_auto_learn": False,
        "loss_clf_flip_scale_p_override": True,
    }
    cfg = {
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_on_closed_candle": True,
        "flip_waive_edge_min": -1.0,
        "flip_seed_waive_edge_min": -0.08,
    }
    assert post_flip_edge_ok(metrics, TradeDirection.PUT, cfg=cfg) is False


def test_gate_seed_candle_blocks_p_ovr_against_closed_candle():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 21
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "tcn_direction": "CALL",
        "calibrated_prob": 0.56,
        "closed_micro_candle_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "scale_vote_call_n": 3,
        "scale_vote_put_n": 0,
        "kelly_fraction_scale": 1.0,
    }
    from unittest.mock import patch

    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.96,
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
            return_value={
                "veto_mode": "soft",
                "veto_p_loss_floor": 0.65,
                "hard_p_loss_floor": 0.90,
                "hard_blocks_pending_waive": True,
                "soft_kelly_mult": 0.55,
                "soft_kelly_mult_high": 0.40,
                "soft_p_loss_high": 0.85,
                "soft_max_stake_pct_high": 0.0025,
                "flip_require_auto_learn": True,
                "flip_allow_seed_on_scale_discord": True,
                "flip_allow_seed_on_cal_discord": True,
                "flip_cal_discord_margin": 0.03,
                "flip_require_pos_edge": True,
                "flip_min_edge_execute": 0.04,
                "flip_waive_on_closed_candle": True,
                "flip_candle_p_loss_floor": 0.85,
                "flip_waive_scale_above_p_loss": 0.95,
                "flip_block_when_tcn_pos_edge": True,
                "flip_waive_edge_min": -1.0,
                "flip_seed_block_against_closed_candle": True,
                "flip_seed_waive_edge_min": -0.08,
            },
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "CALL"
    assert metrics["loss_clf_flip_blocked"] == "seed_candle"
    assert metrics.get("loss_clf_soft") is True


def test_neg_edge_bootstrap_deep_hard_skip():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.56,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": False,
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": False,
                    "neg_edge_soft_when_closed_candle_agree": True,
                    "neg_edge_soft_min_edge": -1.0,
                    "neg_edge_bootstrap_soft_kelly_mult": 0.25,
                    "neg_edge_deep_edge_floor": -0.12,
                }
            }
        },
    }
    orch._log_dedupe = {}
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics.get("neg_edge_bootstrap_deep") is True
    assert metrics["execution_candidate_ready"] is False


def test_neg_edge_uses_fusion_p_eff_avoids_boot_deep_empty():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.55,
        "fusion_applied": True,
        "fusion_p_eff": 0.707,
        "fusion_reason": "ev_put",
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": False,
        "closed_micro_candle_dir": "PUT",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is False
    assert metrics.get("gate_reason") != "neg_edge"
    assert metrics.get("neg_edge_bootstrap_deep") is not True
    assert metrics.get("neg_edge_used_fusion_p_eff") is True
    assert metrics["execution_candidate_ready"] is True
    assert float(metrics["cal_side_edge"]) > 0.04


def test_neg_edge_auto_learn_stays_soft_on_subfloor_edge():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics.get("gate_reason") != "neg_edge"
    assert metrics["neg_edge_soft"] is True
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04
    assert metrics["execution_candidate_ready"] is True
