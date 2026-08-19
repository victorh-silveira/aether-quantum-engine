"""FLIP libera tcn_edge so se tape/vela discordam do TCN."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def _cfg(**overrides):
    base = {
        "veto_mode": "soft",
        "veto_p_loss_floor": 0.65,
        "hard_p_loss_floor": 0.90,
        "hard_blocks_pending_waive": True,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.40,
        "soft_p_loss_high": 0.85,
        "soft_max_stake_pct_high": 0.01,
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_scale_discord": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_cal_discord_margin": 0.03,
        "flip_require_pos_edge": False,
        "flip_min_edge_execute": 0.04,
        "flip_block_when_tcn_pos_edge": True,
        "flip_waive_tcn_pos_edge_on_discord": True,
        "flip_seed_block_against_closed_candle": True,
        "flip_waive_on_closed_candle": False,
        "flip_waive_edge_min": -1.0,
        "flip_seed_waive_edge_min": -0.08,
    }
    base.update(overrides)
    return base


def _orch(cycle_id: int) -> MagicMock:
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = cycle_id
    return orch


def _predict(*, auto: bool):
    return {
        "p_loss": 0.95,
        "veto": True,
        "auto_learn_applied": auto,
        "model_version": "loss_discord",
        "n_train": 40,
        "veto_ready": True,
        "bootstrap": not auto,
    }


def test_tape_discord_allows_flip_despite_tcn_pos_edge():
    orch = _orch(40)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "scale_tape_consensus": "CALL",
        "scale_vote_call_n": 3,
        "scale_vote_put_n": 0,
        "closed_micro_candle_dir": "CALL",
        "ops_window_candle_dir": "CALL",
        "kelly_fraction_scale": 1.0,
        "calibrated_prob": 0.36,
        "raw_prob": 0.36,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=_predict(auto=True),
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_flip") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics.get("loss_clf_flip_tcn_edge_waive_discord") is True
    assert metrics.get("loss_clf_flip_blocked") is None


def test_seed_tape_discord_does_not_pierce_candle_agree():
    orch = _orch(41)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "scale_tape_consensus": "CALL",
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "kelly_fraction_scale": 1.0,
        "calibrated_prob": 0.36,
        "raw_prob": 0.36,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=_predict(auto=False),
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "seed_candle"


def test_tape_and_candle_agree_keeps_tcn_edge_block():
    orch = _orch(42)
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "scale_tape_consensus": "PUT",
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "kelly_fraction_scale": 1.0,
        "calibrated_prob": 0.36,
        "raw_prob": 0.36,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=_predict(auto=True),
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "tcn_pos_edge"
