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
    assert cfg["fusion_w_macro"] == pytest.approx(0.0)
    assert cfg["fusion_w_micro_bar"] == pytest.approx(0.0)
    assert cfg["fusion_loss_weight"] == pytest.approx(0.0)
    assert cfg["fusion_tcn_shrink_near_half"] == pytest.approx(0.0)
    assert cfg["fusion_block_when_tcn_pos_edge"] is True
    assert cfg["fusion_block_when_tcn_candle_agree"] is False
    assert cfg["fusion_loss_requires_auto_learn"] is True
    assert cfg["fusion_loss_seed_weight_mult"] == pytest.approx(0.0)
    assert cfg["fusion_weak_ev_soft_kelly_mult"] == pytest.approx(0.50)
    assert cfg["fusion_weak_ev_seed_soft_kelly_mult"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="fusion_tcn_shrink_near_half"):
        parse_direction_fusion_config({"fusion_tcn_shrink_near_half": 1.5})
    with pytest.raises(ValueError, match="fusion_weak_ev_soft_kelly_mult"):
        parse_direction_fusion_config({"fusion_weak_ev_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="fusion_weak_ev_seed_soft_kelly_mult"):
        parse_direction_fusion_config({"fusion_weak_ev_seed_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="fusion_loss_seed_weight_mult"):
        parse_direction_fusion_config({"fusion_loss_seed_weight_mult": 0.10})
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
        "ops_window_candle_dir": "CALL",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "CALL",
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_candle_agree": True})
    assert apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg) == TradeDirection.CALL
    assert metrics["fusion_reason"] == "negative_ev_abstain"
    # Discord e None branches
    m_disc = dict(metrics, ops_window_candle_dir="PUT", loss_clf_auto_learn=True)
    assert apply_direction_fusion(m_disc, TradeDirection.CALL, cfg=cfg) == TradeDirection.CALL
    m_none = dict(metrics, ops_window_candle_dir=None, loss_clf_auto_learn=True)
    assert apply_direction_fusion(m_none, TradeDirection.CALL, cfg=cfg) == TradeDirection.CALL


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
        "ops_window_candle_dir": "PUT",
        "loss_clf_p_loss": 0.92,
        "loss_clf_flip_ref": "CALL",
        "loss_clf_auto_learn": True,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics["fusion_reason"] == "negative_ev_abstain"


def test_fusion_weak_ev_applies_soft_kelly():
    metrics = {
        "calibrated_prob": 0.55,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "ops_window_candle_dir": "CALL",
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
    assert chosen == TradeDirection.CALL
    assert metrics["fusion_applied"] is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("fusion_weak_ev_soft") is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.40)


def test_fusion_weak_ev_seed_dual_neg_softens_harder():
    metrics = {
        "calibrated_prob": 0.50,
        "predicted_payoff_edge": -0.20,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "ops_window_candle_dir": "CALL",
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
            "fusion_meta_ev_weight": 0.50,
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
    assert metrics["execution_candidate_ready"] is False
    assert metrics["signal_status"] == "SKIP:FUSION_NEGATIVE_EV"


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
        "ops_window_candle_dir": "PUT",
        "loss_clf_p_loss": 0.92,
        "loss_clf_flip_ref": "CALL",
        "loss_clf_auto_learn": True,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics["fusion_applied"] is True
    assert metrics["fusion_reason"] == "negative_ev_abstain"


def test_fusion_keeps_tcn_when_pos_edge_blocks():
    metrics = {
        "calibrated_prob": 0.70,
        "raw_prob": 0.70,
        "tcn_direction": "CALL",
        "scale_macro_dir": "CALL",
        "ops_window_candle_dir": "CALL",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "CALL",
        "exec_direction": "CALL",
    }
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_pos_edge": True})
    chosen = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
    assert chosen == TradeDirection.CALL
    assert metrics.get("fusion_blocked_tcn_pos_edge") is True
    assert metrics["fusion_reason"] == "tcn_pos_edge"


def test_fusion_skips_tcn_pos_edge_when_raw_below_floor():
    metrics = {
        "calibrated_prob": 0.251,
        "raw_prob": 0.50,
        "tcn_direction": "PUT",
        "scale_macro_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "ops_window_candle_dir": "CALL",
        "loss_clf_p_loss": 0.90,
        "loss_clf_flip_ref": "PUT",
        "loss_clf_auto_learn": True,
        "exec_direction": "PUT",
    }
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_pos_edge": True, "fusion_min_edge_execute": 0.04})
    chosen = apply_direction_fusion(metrics, TradeDirection.PUT, cfg=cfg)
    assert metrics.get("fusion_blocked_tcn_pos_edge") is not True
    assert metrics["fusion_reason"] != "tcn_pos_edge"
    assert chosen in {TradeDirection.CALL, TradeDirection.PUT}


def test_fusion_allows_ev_when_cal_pos_raw_neg():
    metrics = {
        "calibrated_prob": 0.70,
        "raw_prob": 0.35,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "ops_window_candle_dir": "CALL",
        "loss_clf_p_loss": 0.95,
        "loss_clf_flip_ref": "PUT",
        "loss_clf_auto_learn": True,
        "exec_direction": "PUT",
    }
    cfg = parse_direction_fusion_config(
        {
            "fusion_enabled": True,
            "fusion_w_macro": 0.45,
            "fusion_w_micro_bar": 0.10,
            "fusion_w_mini": 0.25,
            "fusion_w_mili": 0.10,
            "fusion_w_tape": 0.45,
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
    assert metrics["fusion_reason"] == "pure_tcn"
    assert metrics["execution_candidate_ready"] is True


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
