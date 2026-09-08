from unittest.mock import MagicMock

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def _orch():
    return MagicMock(
        config={"infra": {"loss_classifier": {"enabled": True}}},
        _active_cycle_id=1,
        risk_manager=MagicMock(pending_loss_total=lambda: 0.0, bankroll=1000.0),
        state=MagicMock(balance=1000.0),
    )


def test_loss_clf_hard_skip_above_floor(monkeypatch):
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "calibrated_prob": 0.6,
    }
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        lambda *_a, **_k: {
            "p_loss": 0.91,
            "model_version": "t",
            "n_train": 32,
            "auto_learn_applied": True,
            "veto_ready": True,
        },
    )
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.build_loss_feature_vector",
        lambda *_a, **_k: [0.0] * 24,
    )
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=_orch()) is True
    assert metrics["gate_reason"] == "loss_clf"
    assert metrics["gate_verdict"] == "HARD_SKIP"
    assert metrics["execution_candidate_ready"] is False
    assert metrics["loss_clf_hard"] is True


def test_loss_clf_ok_below_floor(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "calibrated_prob": 0.4,
    }
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        lambda *_a, **_k: {
            "p_loss": 0.89,
            "model_version": "t",
            "n_train": 8,
            "auto_learn_applied": True,
            "veto_ready": True,
        },
    )
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.build_loss_feature_vector",
        lambda *_a, **_k: [0.0] * 24,
    )
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("gate_reason") != "loss_clf"
    assert metrics.get("loss_clf_hard") is not True
    assert metrics.get("execution_candidate_ready") is not False


def test_loss_clf_force_noop():
    metrics = {"execution_candidate_ready": True}
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=_orch(), force=True) is False
