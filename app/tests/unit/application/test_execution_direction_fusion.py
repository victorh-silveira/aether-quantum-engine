"""Testes da fusao EV MACRO/MICRO + Cal/loss/meta."""

from __future__ import annotations

import pytest

from src.application.services.execution_direction_fusion import (
    apply_direction_fusion,
    parse_direction_fusion_config,
)
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import apply_kelly_side_p


def test_parse_direction_fusion_from_ssot():
    cfg = parse_direction_fusion_config({})
    assert cfg["fusion_enabled"] is True
    assert cfg["fusion_replace_adapt_flip"] is True
    assert cfg["fusion_w_macro"] == pytest.approx(0.35)
    assert cfg["fusion_w_micro_bar"] == pytest.approx(0.45)
    assert cfg["fusion_loss_weight"] == pytest.approx(0.80)
    assert cfg["fusion_tcn_shrink_near_half"] == pytest.approx(0.40)
    assert cfg["fusion_block_when_tcn_pos_edge"] is True
    with pytest.raises(ValueError, match="fusion_tcn_shrink_near_half"):
        parse_direction_fusion_config({"fusion_tcn_shrink_near_half": 1.5})


def test_fusion_picks_put_when_tape_and_loss_oppose_weak_call():
    metrics = {
        "calibrated_prob": 0.53,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "PUT",
        "scale_mini_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.92,
        "loss_clf_flip_ref": "CALL",
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.PUT
    assert metrics["fusion_applied"] is True
    assert metrics["fusion_switched"] is True
    assert metrics["fusion_ev_put"] > metrics["fusion_ev_call"]
    assert metrics["fusion_p_eff"] == pytest.approx(metrics["fusion_p_put"])
    assert metrics["execution_candidate_ready"] is True


def test_fusion_keeps_tcn_when_pos_edge_blocks():
    metrics = {
        "calibrated_prob": 0.70,
        "tcn_direction": "CALL",
        "scale_macro_dir": "PUT",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "CALL",
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics.get("fusion_blocked_tcn_pos_edge") is True
    assert metrics["fusion_reason"] == "tcn_pos_edge"


def test_fusion_tie_falls_back_to_cal():
    metrics = {
        "calibrated_prob": 0.55,
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config(
        {
            "fusion_enabled": True,
            "fusion_w_macro": 0.0,
            "fusion_w_micro_bar": 0.0,
            "fusion_w_mini": 0.0,
            "fusion_w_mili": 0.0,
            "fusion_w_tape": 0.0,
            "fusion_meta_ev_weight": 0.0,
            "fusion_loss_weight": 0.0,
            "fusion_tcn_shrink_near_half": 0.0,
            "fusion_block_when_tcn_pos_edge": False,
            "fusion_min_edge_execute": 0.04,
            "fusion_replace_adapt_flip": True,
        }
    )
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics["fusion_reason"] in {"ev_call", "tie_cal"}


def test_fusion_kelly_prefers_fusion_p_call():
    metrics = {
        "fusion_applied": True,
        "fusion_p_call": 0.62,
        "calibrated_prob": 0.51,
        "exec_direction": "CALL",
    }
    p = apply_kelly_side_p(
        metrics,
        order_direction="CALL",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.51,
    )
    assert p >= 0.62 - 1e-9


def test_fusion_then_neg_edge_stays_soft_exec():
    metrics = {
        "calibrated_prob": 0.53,
        "tcn_direction": "CALL",
        "scale_macro_dir": "PUT",
        "closed_micro_candle_dir": "PUT",
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
    assert apply_negative_cal_edge_pause(metrics, orch=None, min_edge=0.04, payout=0.72) is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("gate_reason") != "neg_edge"
    assert metrics.get("neg_edge_soft") is True
