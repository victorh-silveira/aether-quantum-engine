"""Cobertura residual alinhada ao SSOT atual (parte 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    await_post_loss_cooldown,
    log_trading_cycle_cooldown_skip,
    post_loss_cooldown_blocks_trading_cycle,
    schedule_post_loss_cooldown,
)
from src.domain.models.trade import TradeDirection


def test_neg_edge_recovery_and_malformed_paths():
    from src.application.services.execution_neg_edge import (
        _is_recovery_active,
        _min_edge_from_orch,
        _payout_from_orch,
        _signal_skip_raw,
    )

    assert _payout_from_orch(None) == pytest.approx(0.72)
    bad = MagicMock()
    bad.config = "x"
    assert _payout_from_orch(bad) == pytest.approx(0.72)
    assert _is_recovery_active(None) is False
    assert _is_recovery_active(None, {"recovery_mode": True}) is True
    orch = MagicMock()
    orch.risk_manager = None
    assert _is_recovery_active(orch) is False
    orch2 = MagicMock()
    orch2.risk_manager.pending_loss = {"R_10": "x"}
    orch2.risk_manager.consecutive_losses_linear = "bad"
    assert _is_recovery_active(orch2) is False
    orch3 = MagicMock()
    orch3.config = {
        "orchestrator": {"execution": {"signal_skip": {"min_edge_recovery": 0.01, "min_edge_explore": 0.015}}},
        "deep_learning": {},
    }
    orch3.risk_manager.pending_loss = {"R_10": 1.0}
    orch3.risk_manager.consecutive_losses_linear = 0
    assert _min_edge_from_orch(orch3, metrics={"recovery_mode": True}) == pytest.approx(0.01)
    orch_bad_rec = MagicMock()
    orch_bad_rec.config = {
        "orchestrator": {"execution": {"signal_skip": {"min_edge_recovery": object()}}},
        "deep_learning": {"min_edge_execute": 0.04},
    }
    orch_bad_rec.risk_manager.pending_loss = {"R_10": 1.0}
    orch_bad_rec.risk_manager.consecutive_losses_linear = 0
    assert _min_edge_from_orch(orch_bad_rec, metrics={"recovery_mode": True}) == pytest.approx(0.04)
    orch4 = MagicMock()
    orch4.config = {"orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015}}}, "deep_learning": {}}
    orch4.risk_manager.pending_loss = {}
    orch4.risk_manager.consecutive_losses_linear = 0
    assert _min_edge_from_orch(orch4) == pytest.approx(0.015)
    orch_bad_exp = MagicMock()
    orch_bad_exp.config = {
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": object()}}},
        "deep_learning": {"min_edge_execute": 0.04},
    }
    orch_bad_exp.risk_manager.pending_loss = {}
    orch_bad_exp.risk_manager.consecutive_losses_linear = 0
    assert _min_edge_from_orch(orch_bad_exp) == pytest.approx(0.04)
    orch5 = MagicMock()
    orch5.config = {"deep_learning": {"min_edge_execute": "bad"}, "orchestrator": {}}
    orch5.risk_manager.pending_loss = {}
    orch5.risk_manager.consecutive_losses_linear = 0
    assert _min_edge_from_orch(orch5) == 0.0
    assert _signal_skip_raw(None) is None
    orch6 = MagicMock()
    orch6.config = {"orchestrator": "bad"}
    assert _signal_skip_raw(orch6) is None
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.70,
        "edge_zscore": "bad",
    }
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is False


def test_loss_clf_mature_hard_on_flip_block():
    from src.application.services.loss_classifier_gate import apply_loss_classifier_gate

    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "CALL",
        "calibrated_prob": 0.80,
        "raw_prob": 0.80,
        "ops_window_candle_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "scale_macro_dir": "CALL",
        "predicted_payoff_edge": 0.20,
    }
    orch = MagicMock()
    orch.config = {
        "infra": {"loss_classifier": {"enabled": True}},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    orch._active_cycle_id = 1
    orch._log_dedupe = {}
    response = {
        "p_loss": 0.95,
        "veto_ready": True,
        "auto_learn_applied": True,
        "n_train": 40,
        "model_version": "v1",
        "collapsed": False,
    }
    cfg = {
        "veto_mode": "soft",
        "veto_p_loss_floor": 0.65,
        "hard_p_loss_floor": 0.90,
        "hard_blocks_flip_block": True,
        "flip_require_auto_learn": True,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.40,
        "soft_p_loss_high": 0.85,
        "soft_max_stake_pct_high": 0.01,
        "flip_block_when_tcn_pos_edge": True,
        "flip_waive_tcn_pos_edge_on_discord": False,
        "flip_seed_block_against_closed_candle": True,
        "flip_seed_waive_edge_min": -0.08,
        "flip_waive_edge_min": -1.0,
        "flip_candle_p_loss_floor": 0.85,
        "flip_allow_seed_on_scale_discord": False,
        "flip_allow_seed_on_cal_discord": False,
        "flip_require_pos_edge": False,
        "flip_min_edge_execute": 0.04,
        "hard_blocks_pending_waive": True,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=response,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=cfg,
        ),
        patch(
            "src.application.services.loss_classifier_gate.loss_classifier_enabled",
            return_value=True,
        ),
        patch(
            "src.application.services.loss_classifier_gate.build_loss_feature_vector",
            return_value=[0.1] * 8,
        ),
    ):
        blocked = apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="R_10")
    assert blocked is True
    assert metrics.get("gate_reason") == "loss_clf_hard"


def test_post_loss_cooldown_coverage_branches(orch_ready):
    orch = orch_ready
    orch.logger = None
    log_trading_cycle_cooldown_skip(orch)
    orch.logger = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 3
    orch._last_settlement_outcome = "LOSS"
    assert schedule_post_loss_cooldown(orch) == 300.0
    assert post_loss_cooldown_blocks_trading_cycle(orch) is True


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_sleeps_when_active(orch_ready):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 2
    orch._last_settlement_outcome = "LOSS"
    schedule_post_loss_cooldown(orch)
    with patch(
        "src.application.services.orchestrator.post_settlement_loss_cooldown.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleeper:
        rem = await await_post_loss_cooldown(orch)
    assert rem > 0.0
    sleeper.assert_awaited()
