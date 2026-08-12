"""Testes da fusao EV MACRO/MICRO + Cal/loss/meta."""

from __future__ import annotations

import pytest

from src.application.services.execution_direction_fusion import (
    apply_direction_fusion,
    parse_direction_fusion_config,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import apply_kelly_side_p


def test_parse_direction_fusion_from_ssot():
    cfg = parse_direction_fusion_config({})
    assert cfg["fusion_enabled"] is True
    assert cfg["fusion_replace_adapt_flip"] is True
    assert cfg["fusion_w_macro"] == pytest.approx(0.35)
    assert cfg["fusion_w_micro_bar"] == pytest.approx(0.45)
    assert cfg["fusion_loss_weight"] == pytest.approx(0.45)
    assert cfg["fusion_tcn_shrink_near_half"] == pytest.approx(0.25)
    assert cfg["fusion_block_when_tcn_pos_edge"] is True
    assert cfg["fusion_block_when_tcn_candle_agree"] is True
    assert cfg["fusion_loss_requires_auto_learn"] is True
    assert cfg["fusion_loss_seed_weight_mult"] == pytest.approx(0.0)
    assert cfg["fusion_weak_ev_soft_kelly_mult"] == pytest.approx(0.40)
    assert cfg["fusion_weak_ev_seed_soft_kelly_mult"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="fusion_tcn_shrink_near_half"):
        parse_direction_fusion_config({"fusion_tcn_shrink_near_half": 1.5})
    with pytest.raises(ValueError, match="fusion_weak_ev_soft_kelly_mult"):
        parse_direction_fusion_config({"fusion_weak_ev_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="fusion_weak_ev_seed_soft_kelly_mult"):
        parse_direction_fusion_config({"fusion_weak_ev_seed_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="fusion_loss_seed_weight_mult"):
        parse_direction_fusion_config({"fusion_loss_seed_weight_mult": 0.2})


def test_fusion_seed_high_p_loss_keeps_tcn_when_candle_agrees():
    metrics = {
        "calibrated_prob": 0.53,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "PUT",
        "scale_mini_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "closed_micro_candle_dir": "CALL",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "CALL",
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics.get("fusion_blocked_tcn_candle") is True
    assert metrics["fusion_reason"] == "tcn_candle_agree"
    assert metrics["fusion_switched"] is False


def test_fusion_auto_learn_can_switch_when_candle_discords():
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
        "loss_clf_auto_learn": True,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.PUT
    assert metrics.get("fusion_blocked_tcn_candle") is not True
    assert metrics["fusion_switched"] is True


def test_fusion_weak_ev_applies_soft_kelly():
    metrics = {
        "calibrated_prob": 0.51,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "closed_micro_candle_dir": "CALL",
        "loss_clf_p_loss": 0.20,
        "loss_clf_auto_learn": True,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_direction_fusion_config(
        {
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
            "fusion_weak_ev_soft_kelly_mult": 0.40,
        }
    )
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen in (TradeDirection.CALL, TradeDirection.PUT)
    assert metrics["fusion_applied"] is True
    assert float(metrics["fusion_chosen_ev"]) < 0.04
    assert metrics.get("fusion_weak_ev_soft") is True
    assert metrics.get("fusion_weak_ev_seed") is not True
    assert float(metrics["kelly_fraction_scale"]) == pytest.approx(0.40)


def test_fusion_weak_ev_seed_dual_neg_softens_harder():
    metrics = {
        "calibrated_prob": 0.51,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "closed_micro_candle_dir": "CALL",
        "loss_clf_p_loss": 0.20,
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_direction_fusion_config(
        {
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
            "fusion_weak_ev_soft_kelly_mult": 0.40,
            "fusion_weak_ev_seed_soft_kelly_mult": 0.25,
        }
    )
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen in (TradeDirection.CALL, TradeDirection.PUT)
    assert float(metrics["fusion_ev_call"]) < 0.0
    assert float(metrics["fusion_ev_put"]) < 0.0
    assert metrics.get("fusion_weak_ev_seed") is True
    assert float(metrics["kelly_fraction_scale"]) == pytest.approx(0.25)


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
        "loss_clf_auto_learn": True,
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
        "raw_prob": 0.70,
        "tcn_direction": "CALL",
        "scale_macro_dir": "PUT",
        "closed_micro_candle_dir": "PUT",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "CALL",
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_pos_edge": True})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics.get("fusion_blocked_tcn_pos_edge") is True
    assert metrics["fusion_reason"] == "tcn_pos_edge"


def test_fusion_allows_ev_when_cal_pos_raw_neg():
    metrics = {
        "calibrated_prob": 0.24,
        "raw_prob": 0.49,
        "tcn_direction": "PUT",
        "scale_macro_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "closed_micro_candle_dir": "CALL",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "PUT",
        "loss_clf_auto_learn": True,
        "exec_direction": "PUT",
    }
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_pos_edge": True})
    chosen = apply_direction_fusion(metrics, TradeDirection.PUT, cfg=cfg)
    assert metrics.get("fusion_blocked_tcn_pos_edge") is not True
    assert metrics.get("loss_clf_flip_cal_raw_discord") is True
    assert chosen == TradeDirection.CALL
    assert metrics["fusion_reason"] == "ev_call"


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


def test_fusion_kelly_prefers_fusion_p_eff():
    metrics = {
        "fusion_applied": True,
        "fusion_p_call": 0.55,
        "fusion_p_eff": 0.90,
        "calibrated_prob": 0.51,
        "exec_direction": "PUT",
    }
    p = apply_kelly_side_p(
        metrics,
        order_direction="PUT",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.51,
        payout=0.72,
    )
    assert p == pytest.approx(0.90)
    assert metrics.get("kelly_used_fusion_p_eff") is True
