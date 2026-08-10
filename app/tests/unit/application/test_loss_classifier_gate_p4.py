"""Testes do loss-classifier (parte 4): gate auxiliar e bankroll."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def _soft_cfg(**overrides):
    base = {
        "veto_mode": "soft",
        "veto_p_loss_floor": 0.65,
        "hard_p_loss_floor": 0.90,
        "hard_blocks_pending_waive": True,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.20,
        "soft_p_loss_high": 0.85,
        "soft_max_stake_pct_high": 0.0025,
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_scale_discord": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_cal_discord_margin": 0.03,
        "flip_require_pos_edge": False,
        "flip_min_edge_execute": 0.04,
    }
    base.update(overrides)
    return base


def test_resolve_soft_kelly_mult_graduated():
    from src.application.services.loss_classifier_gate import resolve_soft_kelly_mult

    cfg = {
        "veto_p_loss_floor": 0.65,
        "soft_p_loss_high": 0.85,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.20,
    }
    assert resolve_soft_kelly_mult(0.50, cfg) == pytest.approx(0.55)
    assert resolve_soft_kelly_mult(0.85, cfg) == pytest.approx(0.20)
    assert resolve_soft_kelly_mult(0.75, cfg) == pytest.approx(0.375)


def test_block_reason_ignores_loss_clf_veto():
    from src.application.services.orchestrator.execution_blockers import _candidate_block_reason

    assert _candidate_block_reason({"gate_reason": "loss_clf_veto"}) is None
    assert _candidate_block_reason({"gate_reason": "training"}) == "training"


def test_feed_loss_classifier_learn_paths():
    from src.application.services.orchestrator.settlement_outcome import _feed_loss_classifier_learn

    orch = MagicMock()
    orch._loss_clf_vectors = None
    _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=1)
    orch._loss_clf_vectors = {"R_10": []}
    _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=1)
    orch._loss_clf_vectors = {"R_10": [0.1] * 24}
    orch.config = None
    _feed_loss_classifier_learn(orch, "R_10", won=False, contract_id=2)
    orch._loss_clf_vectors = {"R_10": [0.2] * 24, "cid:9": [0.3] * 24}
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
        return_value={"ok": True, "buffer_n": 12, "retrained": True, "n_train": 12},
    ) as learn:
        _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=9)
        learn.assert_called_once()
        assert learn.call_args.kwargs["label"] == "WIN"
        assert learn.call_args.kwargs["symbol"] == "R_10"
        assert learn.call_args.kwargs["feature_vector"] == [0.3] * 24
    assert "R_10" not in orch._loss_clf_vectors
    assert "cid:9" not in orch._loss_clf_vectors


def test_loss_feature_vector_store_bind_and_pop():
    from src.application.services.loss_classifier_vectors import (
        bind_loss_feature_vector_to_contract,
        pop_loss_feature_vector,
        store_loss_feature_vector,
    )

    orch = MagicMock()
    orch._loss_clf_vectors = None
    store_loss_feature_vector(orch, "R_10", [0.5] * 24)
    bind_loss_feature_vector_to_contract(orch, "R_10", 42)
    assert orch._loss_clf_vectors["cid:42"] == [0.5] * 24
    assert pop_loss_feature_vector(orch, "R_10", 42) == [0.5] * 24
    assert pop_loss_feature_vector(orch, "R_10", 99) is None


def test_features_bankroll_norm_and_dim_guard():
    from src.application.services.loss_classifier_features import build_loss_feature_vector

    vector = build_loss_feature_vector(
        {"direction_margin": 0.02},
        TradeDirection.CALL,
        pending=250.0,
        linear=0,
        bankroll=10000.0,
    )
    assert vector[9] == pytest.approx(0.025)
    with (
        patch("src.application.services.loss_classifier_features.LOSS_FEATURE_DIM", 99),
        pytest.raises(ValueError, match="loss feature dim"),
    ):
        build_loss_feature_vector({}, TradeDirection.CALL)


def test_gate_logs_ok_for_bootstrap_ready():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 3
    payload = {
        "p_loss": 0.42,
        "veto": False,
        "auto_learn_applied": False,
        "model_version": "loss_bootstrap_synth",
        "n_train": 64,
        "veto_ready": True,
        "bootstrap": True,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert "loss_clf_ok:3" in orch._log_dedupe
    assert "loss_clf_cold:3" not in orch._log_dedupe


def test_gate_bankroll_falls_back_to_risk_manager():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=100.0)
    orch.risk_manager.bankroll = 2000.0
    orch.state.balance = 0.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 9
    captured: dict[str, float] = {}

    def _capture_vector(metrics, exec_dir, *, pending=0.0, linear=0, bankroll=0.0):
        captured["bankroll"] = float(bankroll)
        captured["pending"] = float(pending)
        return [0.0] * 24

    payload = {
        "p_loss": 0.40,
        "veto": False,
        "auto_learn_applied": False,
        "model_version": "v",
        "n_train": 40,
        "veto_ready": True,
        "bootstrap": False,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.build_loss_feature_vector",
            side_effect=_capture_vector,
        ),
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert captured["bankroll"] == pytest.approx(2000.0)
    assert captured["pending"] == pytest.approx(100.0)


def test_gate_bankroll_zero_when_balance_and_risk_invalid():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.risk_manager.bankroll = object()
    orch.state.balance = object()
    orch._log_dedupe = {}
    orch._active_cycle_id = 11
    captured: dict[str, float] = {}

    def _capture_vector(metrics, exec_dir, *, pending=0.0, linear=0, bankroll=0.0):
        captured["bankroll"] = float(bankroll)
        return [0.0] * 24

    payload = {
        "p_loss": 0.40,
        "veto": False,
        "auto_learn_applied": False,
        "model_version": "v",
        "n_train": 40,
        "veto_ready": True,
        "bootstrap": False,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.build_loss_feature_vector",
            side_effect=_capture_vector,
        ),
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert captured["bankroll"] == pytest.approx(0.0)


def test_loss_clf_clears_stale_and_dedupes_per_cycle():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    payload = {
        "p_loss": 0.9178,
        "veto": True,
        "auto_learn_applied": True,
        "model_version": "loss_live_n40",
        "n_train": 40,
        "veto_ready": True,
        "bootstrap": False,
    }
    metrics = {
        "kelly_fraction_scale": 1.0,
        "execution_candidate_ready": True,
        "loss_clf_hard": True,
        "gate_reason": "loss_clf_hard",
        "signal_status": "SKIP:LOSS_CLF_HARD",
        "loss_clf_p_loss": 0.99,
        "exec_direction": "CALL",
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "scale_vote_call_n": 1,
        "scale_vote_put_n": 1,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value=_soft_cfg(),
        ),
    ):
        orch._active_cycle_id = 1
        assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
        assert metrics["loss_clf_cycle_id"] == 1
        assert metrics["loss_clf_p_loss"] == pytest.approx(0.9178)
        assert metrics["exec_direction"] == "PUT"
        assert "loss_clf_flip:1" in orch._log_dedupe
        orch._active_cycle_id = 2
        assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
        assert metrics["loss_clf_cycle_id"] == 2
        assert "loss_clf_flip:2" in orch._log_dedupe
        assert orch._log_dedupe["loss_clf_flip:1"] == orch._log_dedupe["loss_clf_flip:2"]
