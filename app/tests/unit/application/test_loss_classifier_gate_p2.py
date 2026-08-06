"""Testes do loss-classifier (parte 2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.domain.models.trade import TradeDirection


def test_apply_loss_classifier_veto_and_ok_paths():
    metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "scale_adapted": True,
        "scale_micro_regime": "chop",
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "kelly_fraction_scale": 1.0,
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = "bad"
    orch._log_dedupe = {}
    veto_payload = {
        "p_loss": 0.8,
        "veto": True,
        "auto_learn_applied": True,
        "model_version": "auto1",
        "n_train": 40,
        "veto_ready": True,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=veto_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value={
                "veto_mode": "soft",
                "veto_p_loss_floor": 0.65,
                "soft_kelly_mult": 0.55,
                "soft_kelly_mult_high": 0.20,
                "soft_p_loss_high": 0.85,
                "soft_max_stake_pct_high": 0.0025,
            },
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="OTC_SPC") is False
    assert metrics.get("gate_reason") is None
    assert metrics["loss_clf_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.2875)
    soft_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "scale_micro_regime": "chop",
    }
    soft_payload = dict(veto_payload)
    soft_payload["p_loss"] = 0.65
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=soft_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value={
                "veto_mode": "soft",
                "veto_p_loss_floor": 0.65,
                "soft_kelly_mult": 0.55,
                "soft_kelly_mult_high": 0.20,
                "soft_p_loss_high": 0.85,
                "soft_max_stake_pct_high": 0.0025,
            },
        ),
    ):
        assert apply_loss_classifier_gate(soft_metrics, TradeDirection.PUT, orch=orch, symbol="OTC_SPC") is False
    assert soft_metrics.get("gate_reason") is None
    assert soft_metrics.get("execution_candidate_ready") is not False
    assert soft_metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert soft_metrics["loss_clf_soft"] is True
    assert soft_metrics["loss_clf_soft_max_stake_pct"] == pytest.approx(0.0025)
    pending_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
    }
    orch.risk_manager.pending_loss_total = MagicMock(return_value=38.0)
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=dict(veto_payload, p_loss=0.90),
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value={
                "veto_mode": "soft",
                "veto_p_loss_floor": 0.65,
                "soft_kelly_mult": 0.55,
                "soft_kelly_mult_high": 0.20,
                "soft_p_loss_high": 0.85,
                "soft_max_stake_pct_high": 0.0025,
            },
        ),
    ):
        assert apply_loss_classifier_gate(pending_metrics, TradeDirection.CALL, orch=orch, symbol="OTC_SPC") is False
    assert pending_metrics["loss_clf_soft"] is True
    assert pending_metrics.get("loss_clf_soft_waived_pending") is True
    assert pending_metrics["kelly_fraction_scale"] == pytest.approx(1.0)
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    high_metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "kelly_fraction_scale": 1.0,
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
        "scale_micro_regime": "chop",
    }
    high_payload = dict(veto_payload)
    high_payload["p_loss"] = 0.90
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value=high_payload,
        ),
        patch(
            "src.application.services.loss_classifier_gate.resolve_loss_classifier_config",
            return_value={
                "veto_mode": "soft",
                "veto_p_loss_floor": 0.65,
                "soft_kelly_mult": 0.55,
                "soft_kelly_mult_high": 0.20,
                "soft_p_loss_high": 0.85,
                "soft_max_stake_pct_high": 0.0025,
            },
        ),
    ):
        assert apply_loss_classifier_gate(high_metrics, TradeDirection.CALL, orch=orch, symbol="OTC_SPC") is False
    assert high_metrics["kelly_fraction_scale"] == pytest.approx(0.20)
    assert high_metrics["loss_clf_soft_max_stake_pct"] == pytest.approx(0.0025)
    assert high_metrics["loss_clf_soft_kelly_mult"] == pytest.approx(0.20)
    metrics2 = dict(metrics)
    metrics2.pop("gate_reason", None)
    metrics2.pop("signal_skip_reason", None)
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.2,
            "veto": False,
            "auto_learn_applied": False,
            "model_version": "m",
            "n_train": 5,
            "veto_ready": False,
        },
    ):
        assert apply_loss_classifier_gate(metrics2, TradeDirection.PUT, orch=orch, symbol="OTC_SPC") is False


def test_skip_before_loss_and_force_and_disabled():
    assert apply_loss_classifier_gate({"gate_reason": "cal_margin"}, TradeDirection.CALL, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, force=True, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=None) is False
    with patch("src.application.services.loss_classifier_gate.loss_classifier_enabled", return_value=False):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=MagicMock()) is False


def test_resolve_soft_kelly_mult_graduated():
    from src.application.services.loss_classifier_gate import resolve_soft_kelly_mult

    cfg = {
        "veto_p_loss_floor": 0.65,
        "soft_p_loss_high": 0.85,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.20,
    }
    assert resolve_soft_kelly_mult(0.65, cfg) == pytest.approx(0.55)
    assert resolve_soft_kelly_mult(0.85, cfg) == pytest.approx(0.20)
    assert resolve_soft_kelly_mult(0.90, cfg) == pytest.approx(0.20)
    mid = resolve_soft_kelly_mult(0.75, cfg)
    assert mid == pytest.approx(0.375)


def test_block_reason_ignores_loss_clf_veto():
    assert _candidate_block_reason({"gate_reason": "loss_clf_veto"}) is None
    assert _candidate_block_reason({"gate_reason": "training"}) == "training"


def test_feed_loss_classifier_learn_paths():
    from src.application.services.orchestrator.settlement_outcome import _feed_loss_classifier_learn

    orch = MagicMock()
    orch._loss_clf_vectors = None
    _feed_loss_classifier_learn(orch, "OTC_SPC", won=True, contract_id=1)
    orch._loss_clf_vectors = {"OTC_SPC": []}
    _feed_loss_classifier_learn(orch, "OTC_SPC", won=True, contract_id=1)
    orch._loss_clf_vectors = {"OTC_SPC": [0.1] * 24}
    orch.config = None
    _feed_loss_classifier_learn(orch, "OTC_SPC", won=False, contract_id=2)
    orch._loss_clf_vectors = {"OTC_SPC": [0.2] * 24, "cid:9": [0.3] * 24}
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
        return_value={"ok": True, "buffer_n": 12, "retrained": True, "n_train": 12},
    ) as learn:
        _feed_loss_classifier_learn(orch, "OTC_SPC", won=True, contract_id=9)
        learn.assert_called_once()
        assert learn.call_args.kwargs["label"] == "WIN"
        assert learn.call_args.kwargs["symbol"] == "OTC_SPC"
        assert learn.call_args.kwargs["feature_vector"] == [0.3] * 24
    assert "OTC_SPC" not in orch._loss_clf_vectors
    assert "cid:9" not in orch._loss_clf_vectors


def test_loss_feature_vector_store_bind_and_pop():
    from src.application.services.loss_classifier_vectors import (
        bind_loss_feature_vector_to_contract,
        pop_loss_feature_vector,
        store_loss_feature_vector,
    )

    orch = MagicMock()
    orch._loss_clf_vectors = None
    store_loss_feature_vector(orch, "OTC_SPC", [0.5] * 24)
    bind_loss_feature_vector_to_contract(orch, "OTC_SPC", 42)
    assert orch._loss_clf_vectors["cid:42"] == [0.5] * 24
    assert pop_loss_feature_vector(orch, "OTC_SPC", 42) == [0.5] * 24
    assert pop_loss_feature_vector(orch, "OTC_SPC", 99) is None


def test_features_bankroll_norm_and_dim_guard():
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
