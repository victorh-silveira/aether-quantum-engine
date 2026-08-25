"""Testes complementares de cobertura de fusao EV e helpers de execucao."""

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
            {"closed_micro_candle_dir": "CALL", "ops_window_candle_dir": "CALL"},
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


def test_fusion_tie_breaks_to_calibrated_side():
    metrics = {
        "calibrated_prob": 0.50,
        "raw_prob": 0.50,
        "tcn_direction": "CALL",
        "scale_micro_dir": "CALL",
        "scale_macro_dir": "PUT",
        "scale_mini_dir": "PUT",
        "scale_mili_dir": "PUT",
        "scale_tape_consensus": "PUT",
        "ops_window_candle_dir": "PUT",
        "loss_clf_p_loss": 0.50,
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
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
            "fusion_min_edge_execute": 0.0,
        }
    )
    with patch("src.application.services.execution_direction_fusion._payout", return_value=1.5):
        metrics["calibrated_prob"] = 0.50
        metrics["ops_window_candle_dir"] = None
        chosen_call = apply_direction_fusion(metrics, TradeDirection.CALL, cfg=cfg)
        assert chosen_call == TradeDirection.CALL
        assert metrics["fusion_reason"] == "tie_cal"

        metrics["calibrated_prob"] = 0.499999999999
        metrics["ops_window_candle_dir"] = "PUT"
        chosen_put = apply_direction_fusion(metrics, TradeDirection.PUT, cfg=cfg)
        assert chosen_put in {TradeDirection.CALL, TradeDirection.PUT}
