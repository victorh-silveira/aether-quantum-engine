"""Fusao EV + neg_edge / seed (parte 3)."""

from __future__ import annotations

from src.application.services.execution_direction_fusion import (
    apply_direction_fusion,
    parse_direction_fusion_config,
)
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.domain.models.trade import TradeDirection


def test_fusion_then_neg_edge_hard_when_cal_nonpositive():
    metrics = {
        "calibrated_prob": 0.53,
        "tcn_direction": "CALL",
        "scale_macro_dir": "PUT",
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "loss_clf_p_loss": 0.91,
        "loss_clf_flip_ref": "CALL",
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.PUT
    paused = apply_negative_cal_edge_pause(metrics, orch=None, min_edge=0.04, payout=0.72)
    assert paused is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("gate_reason") != "neg_edge"
    assert metrics.get("neg_edge_fusion_waived") is True
    assert float(metrics["cal_side_edge"]) <= 0.0


def test_fusion_put_seed_empty_when_cal_nonpositive():
    metrics = {
        "calibrated_prob": 0.55,
        "raw_prob": 0.58,
        "tcn_direction": "CALL",
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "scale_macro_dir": "PUT",
        "loss_clf_p_loss": 0.90,
        "loss_clf_flip_ref": "CALL",
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.PUT
    assert float(metrics["fusion_p_eff"]) > 0.58
    orch = type(
        "O",
        (),
        {
            "config": {
                "risk_management": {"params": {"payout_estimate": 0.72}},
                "deep_learning": {"min_edge_execute": 0.04},
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
        },
    )()
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("gate_reason") != "neg_edge"
    assert metrics.get("neg_edge_fusion_waived") is True
