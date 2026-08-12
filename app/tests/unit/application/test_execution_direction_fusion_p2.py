"""Testes da fusao EV (parte 2): branches e cobertura de helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_direction_fusion import (
    apply_direction_fusion,
    parse_direction_fusion_config,
)
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause, parse_neg_edge_soft_config
from src.application.services.execution_signal_skip import parse_signal_skip_config
from src.application.services.loss_classifier_flip import seed_candle_blocks_flip
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import _read_call_prob
from src.infrastructure.inference.loss_classifier_client import resolve_loss_classifier_config


def test_parse_fusion_rejects_weight_and_min_edge():
    with pytest.raises(ValueError, match="fusion_w_macro"):
        parse_direction_fusion_config({"fusion_w_macro": 3.0})
    with pytest.raises(ValueError, match="fusion_min_edge_execute"):
        parse_direction_fusion_config({"fusion_min_edge_execute": 0.9})


def test_fusion_disabled_and_no_cal_and_meta_bad():
    metrics = {"calibrated_prob": 0.55, "exec_direction": "CALL"}
    disabled = parse_direction_fusion_config({"fusion_enabled": False})
    assert apply_direction_fusion(metrics, TradeDirection.CALL, cfg=disabled) == TradeDirection.CALL
    assert metrics["fusion_reason"] == "disabled"
    bare = {"exec_direction": "PUT"}
    cfg = parse_direction_fusion_config({"fusion_block_when_tcn_pos_edge": False})
    assert apply_direction_fusion(bare, TradeDirection.PUT, cfg=cfg) == TradeDirection.PUT
    assert bare["fusion_reason"] == "no_cal"
    metrics2 = {
        "calibrated_prob": 0.55,
        "exec_direction": "CALL",
        "predicted_payoff_edge": object(),
    }
    cfg2 = parse_direction_fusion_config(
        {
            "fusion_block_when_tcn_pos_edge": False,
            "fusion_tcn_shrink_near_half": 0.0,
            "fusion_w_macro": 0.0,
            "fusion_w_micro_bar": 0.0,
            "fusion_w_mini": 0.0,
            "fusion_w_mili": 0.0,
            "fusion_w_tape": 0.0,
            "fusion_loss_weight": 0.0,
            "fusion_meta_ev_weight": 0.5,
        }
    )
    assert apply_direction_fusion(metrics2, TradeDirection.CALL, cfg=cfg2) == TradeDirection.CALL
    metrics3 = {
        "calibrated_prob": 0.55,
        "loss_clf_p_loss": "bad",
        "loss_clf_flip_ref": "CALL",
        "exec_direction": "CALL",
    }
    cfg3 = parse_direction_fusion_config(
        {
            "fusion_block_when_tcn_pos_edge": False,
            "fusion_tcn_shrink_near_half": 0.0,
            "fusion_w_macro": 0.0,
            "fusion_w_micro_bar": 0.0,
            "fusion_w_mini": 0.0,
            "fusion_w_mili": 0.0,
            "fusion_w_tape": 0.0,
            "fusion_loss_weight": 0.8,
            "fusion_meta_ev_weight": 0.0,
        }
    )
    assert apply_direction_fusion(metrics3, TradeDirection.CALL, cfg=cfg3) == TradeDirection.CALL
    metrics4 = {"calibrated_prob": 0.55, "loss_clf_p_loss": 0.9, "exec_direction": "CALL"}
    assert apply_direction_fusion(metrics4, TradeDirection.CALL, cfg=cfg3) == TradeDirection.CALL


def test_fusion_raw_and_payout_helpers():
    from src.application.services import execution_direction_fusion as mod

    assert mod._fusion_raw(None) is None
    assert mod._fusion_raw(SimpleNamespace(config="x")) is None
    assert mod._fusion_raw(SimpleNamespace(config={"orchestrator": "x"})) is None
    assert mod._fusion_raw(SimpleNamespace(config={"orchestrator": {"execution": "x"}})) is None
    assert mod._fusion_raw(SimpleNamespace(config={"orchestrator": {"execution": {}}})) is None
    raw = mod._fusion_raw(
        SimpleNamespace(config={"orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}}})
    )
    assert raw == {"fusion_enabled": False}
    assert mod._payout(None) == pytest.approx(0.72)
    assert mod._payout(SimpleNamespace(config="x")) == pytest.approx(0.72)
    assert mod._payout(
        SimpleNamespace(config={"risk_management": {"params": {"payout_estimate": "x"}}})
    ) == pytest.approx(0.72)
    assert mod._payout(
        SimpleNamespace(config={"risk_management": {"params": {"payout_estimate": 0.8}}})
    ) == pytest.approx(0.8)
    assert mod._sigmoid(40.0) > 0.99
    assert mod._sigmoid(-40.0) < 0.01
    assert mod._shrink_near_half(0.6, 0.0) == pytest.approx(0.6)
    assert mod._read_p_call({"calibrated_prob": object()}) is None
    assert mod._read_p_call({"raw_prob": 0.4}) == pytest.approx(0.4)
    assert mod._loss_logit_bonus({"loss_clf_p_loss": None}, "CALL", {"fusion_loss_weight": 0.8}) == 0.0
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": 0.9, "loss_clf_flip_ref": "CALL", "loss_clf_auto_learn": True},
            "CALL",
            {"fusion_loss_weight": 0.8, "fusion_loss_requires_auto_learn": True},
        )
        < 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": 0.9, "loss_clf_flip_ref": "CALL", "loss_clf_auto_learn": True},
            "PUT",
            {"fusion_loss_weight": 0.8, "fusion_loss_requires_auto_learn": True},
        )
        > 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": 0.95, "loss_clf_flip_ref": "CALL", "loss_clf_auto_learn": False},
            "PUT",
            {
                "fusion_loss_weight": 0.8,
                "fusion_loss_requires_auto_learn": True,
                "fusion_loss_seed_weight_mult": 0.0,
            },
        )
        == 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": 0.95, "loss_clf_flip_ref": "CALL", "loss_clf_auto_learn": False},
            "PUT",
            {
                "fusion_loss_weight": 0.8,
                "fusion_loss_requires_auto_learn": True,
                "fusion_loss_seed_weight_mult": 0.10,
            },
        )
        > 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": "bad", "loss_clf_flip_ref": "CALL", "loss_clf_auto_learn": True},
            "PUT",
            {"fusion_loss_weight": 0.8, "fusion_loss_requires_auto_learn": True},
        )
        == 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {"loss_clf_p_loss": 0.9, "loss_clf_auto_learn": True},
            "PUT",
            {"fusion_loss_weight": 0.8, "fusion_loss_requires_auto_learn": True},
        )
        == 0.0
    )
    assert (
        mod._loss_logit_bonus(
            {
                "loss_clf_p_loss": 0.95,
                "loss_clf_flip_ref": "CALL",
                "loss_clf_auto_learn": True,
                "loss_clf_collapsed": True,
            },
            "PUT",
            {"fusion_loss_weight": 0.8, "fusion_loss_requires_auto_learn": True},
        )
        == 0.0
    )
    stay = {"calibrated_prob": 0.7, "tcn_direction": "CALL", "exec_direction": "CALL"}
    cfg_stay = parse_direction_fusion_config(
        {
            "fusion_block_when_tcn_pos_edge": False,
            "fusion_w_macro": 0.0,
            "fusion_w_micro_bar": 0.0,
            "fusion_w_mini": 0.0,
            "fusion_w_mili": 0.0,
            "fusion_w_tape": 0.0,
            "fusion_loss_weight": 0.0,
            "fusion_meta_ev_weight": 0.0,
            "fusion_tcn_shrink_near_half": 0.0,
        }
    )
    assert apply_direction_fusion(stay, TradeDirection.CALL, cfg=cfg_stay) == TradeDirection.CALL
    assert stay.get("fusion_switched") is False
    candle_off = {
        "calibrated_prob": 0.55,
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "closed_micro_candle_side": "CALL",
        "scale_micro_dir": "PUT",
        "scale_macro_dir": "PUT",
        "scale_mini_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_dir": "PUT",
    }
    cfg_candle_off = parse_direction_fusion_config(
        {
            "fusion_block_when_tcn_pos_edge": False,
            "fusion_block_when_tcn_candle_agree": False,
            "fusion_tcn_shrink_near_half": 0.0,
            "fusion_min_edge_execute": 0.0,
        }
    )
    with patch(
        "src.application.services.execution_direction_fusion.closed_micro_candle_side",
        return_value="CALL",
    ):
        out = apply_direction_fusion(candle_off, TradeDirection.CALL, cfg=cfg_candle_off)
    assert out in {TradeDirection.CALL, TradeDirection.PUT}
    assert candle_off.get("fusion_reason") != "tcn_candle_agree"


def test_precommit_cov_gaps_startup_neg_flip_meta():
    from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop

    with patch(
        "src.application.services.deep_learning.dl_startup.all_symbols_have_checkpoints",
        return_value=True,
    ):
        orch = SimpleNamespace(
            symbols=["R_10"],
            config={
                "deep_learning": {"online_training": True},
                "orchestrator": {"engine_mode": "train"},
                "data_handler": {},
            },
        )
        assert prepare_inference_run_loop(orch) is False
    with pytest.raises(ValueError, match="neg_edge_bootstrap_soft_kelly_mult"):
        parse_neg_edge_soft_config({"neg_edge_bootstrap_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="neg_edge_deep_edge_floor"):
        parse_neg_edge_soft_config({"neg_edge_deep_edge_floor": 0.1})
    with pytest.raises(ValueError, match="neg_edge_bootstrap_soft_kelly_mult"):
        parse_signal_skip_config({"neg_edge_bootstrap_soft_kelly_mult": 1.5})
    with pytest.raises(ValueError, match="neg_edge_deep_edge_floor"):
        parse_signal_skip_config({"neg_edge_deep_edge_floor": -1.5})
    assert (
        seed_candle_blocks_flip(
            {"closed_micro_candle_dir": "CALL"},
            {"auto_learn_applied": False},
            TradeDirection.CALL,
            cfg={"flip_seed_block_against_closed_candle": False},
        )
        is False
    )
    with pytest.raises(ValueError, match="flip_seed_waive_edge_min"):
        resolve_loss_classifier_config({"flip_seed_waive_edge_min": 0.1})
    assert _read_call_prob({"fusion_applied": True, "fusion_p_call": object()}) is None
    assert _read_call_prob({"fusion_applied": True, "fusion_p_call": 0.66}) == pytest.approx(0.66)
    assert (
        apply_negative_cal_edge_pause(
            {
                "execution_candidate_ready": True,
                "exec_direction": "CALL",
                "calibrated_prob": 0.4,
                "loss_clf_auto_learn": True,
            },
            orch=SimpleNamespace(config="bad"),
            min_edge=0.04,
            payout=0.72,
            soft_mult=0.55,
        )
        is True
    )


@pytest.mark.asyncio
async def test_learn_meta_via_config_sync_with_running_loop():
    from src.infrastructure.inference.meta_classifier_pool import learn_meta_via_config_sync

    with patch(
        "src.infrastructure.inference.meta_classifier_pool.get_meta_classifier_client",
        new_callable=AsyncMock,
    ) as mock_get:
        client = MagicMock()
        client.learn = AsyncMock(return_value={"ok": True, "retrained": True})
        mock_get.return_value = client
        out = learn_meta_via_config_sync(
            {"infra": {"meta_classifier": {"enabled": True, "online_learn": True, "timeout_seconds": 1.0}}},
            feature_vector=[0.0] * 43,
            target=0.1,
        )
    assert out["retrained"] is True
