"""FLIP seed com vela discord + soft neg_edge pos-FLIP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.loss_classifier_flip import flip_reason_token, resolve_flip_waivers
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def test_resolve_flip_waivers_seed_candle_discord():
    metrics: dict = {"closed_micro_candle_dir": "CALL"}
    response = {"auto_learn_applied": False}
    cfg = {
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_waive_on_closed_candle": False,
        "flip_waive_scale_above_p_loss": 1.01,
        "hard_p_loss_floor": 0.90,
        "flip_allow_seed_on_candle_discord": True,
    }
    seed, scale = resolve_flip_waivers(metrics, response, TradeDirection.PUT, cfg=cfg, p_loss=0.958)
    assert seed is False
    assert scale is False
    assert metrics["loss_clf_flip_seed_candle_discord"] is True
    assert flip_reason_token(metrics) == "seed_candle"


def test_resolve_flip_waivers_seed_stays_blocked_without_candle_or_low_p():
    metrics = {"closed_micro_candle_dir": "PUT"}
    response = {"auto_learn_applied": False}
    cfg = {
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_candle_discord": True,
        "hard_p_loss_floor": 0.90,
        "flip_waive_on_closed_candle": False,
        "flip_waive_scale_above_p_loss": 1.01,
    }
    seed, _ = resolve_flip_waivers(metrics, response, TradeDirection.PUT, cfg=cfg, p_loss=0.958)
    assert seed is True
    metrics2 = {"closed_micro_candle_dir": "CALL"}
    seed2, _ = resolve_flip_waivers(metrics2, response, TradeDirection.PUT, cfg=cfg, p_loss=0.896)
    assert seed2 is True


def test_gate_seed_flips_when_candle_discords():
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
        "calibrated_prob": 0.346,
        "raw_prob": 0.30,
        "closed_micro_candle_dir": "CALL",
        "kelly_fraction_scale": 1.0,
    }
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.95783,
            "model_version": "seed-test",
            "n_train": 64,
            "auto_learn_applied": False,
            "veto_ready": True,
            "collapsed": False,
        },
    ):
        apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="1HZ75V")
    assert metrics.get("loss_clf_flip") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics.get("loss_clf_flip_seed_candle_discord") is True
    assert metrics.get("loss_clf_flip_blocked") in (None, "")


def test_neg_edge_soft_after_loss_flip_candle_agree():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.346,
        "kelly_fraction_scale": 1.0,
        "loss_clf_flip": True,
        "closed_micro_candle_dir": "CALL",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.015, "min_edge_explore": 0.015},
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
    assert metrics["execution_candidate_ready"] is True
    assert metrics["neg_edge_soft"] is True
    assert metrics["neg_edge_candle_soft"] is True
    assert metrics["gate_verdict"] == "SOFT_SIZE"
    assert metrics["signal_skip_waived"] == "neg_edge_soft"


def test_neg_edge_soft_flip_subfloor_candle_agree():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.545,
        "kelly_fraction_scale": 1.0,
        "loss_clf_flip": True,
        "closed_micro_candle_dir": "CALL",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04, "min_edge_explore": 0.04},
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
    assert metrics["neg_edge_candle_soft"] is True
    assert metrics["gate_verdict"] == "SOFT_SIZE"
    assert float(metrics["cal_side_edge"]) > 0.0
    assert float(metrics["cal_side_edge"]) < 0.04


def test_neg_edge_hard_without_flip_even_if_candle_agrees():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.346,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "CALL",
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
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics["gate_verdict"] == "HARD_SKIP"
