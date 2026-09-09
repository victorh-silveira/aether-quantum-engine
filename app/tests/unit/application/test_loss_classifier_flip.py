from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.loss_classifier_gate_support import compute_loss_clf_p_eff
from src.domain.models.trade import TradeDirection


def _orch():
    return MagicMock(
        config={"infra": {"loss_classifier": {"enabled": True}}},
        _active_cycle_id=1,
        risk_manager=MagicMock(pending_loss_total=lambda: 0.0, bankroll=1000.0),
        state=MagicMock(balance=1000.0),
    )


def _patch_predict(
    monkeypatch,
    *,
    p_loss: float,
    n_train: int,
    auto_learn: bool = True,
    bootstrap: bool = False,
):
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        lambda *_a, **_k: {
            "p_loss": p_loss,
            "model_version": "t",
            "n_train": n_train,
            "auto_learn_applied": auto_learn,
            "veto_ready": True,
            "bootstrap": bootstrap,
        },
    )
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.build_loss_feature_vector",
        lambda *_a, **_k: [0.0] * 24,
    )


def test_compute_p_eff_young_floor_055():
    p_eff, young, discord, floor = compute_loss_clf_p_eff(
        0.10,
        n_train=12,
        flip_trust_n=32,
        flip_young_shrink=0.35,
        hard_floor=0.55,
        flip_young_p_eff_floor=0.55,
        tcn_ref=TradeDirection.PUT,
        tape="CALL",
    )
    assert young is True
    assert discord is True
    assert p_eff == 0.5 + (0.10 - 0.5) * 0.35
    assert floor == 0.55


def test_compute_p_eff_mature_uses_hard_floor():
    p_eff, young, discord, floor = compute_loss_clf_p_eff(
        0.10,
        n_train=64,
        flip_trust_n=32,
        flip_young_shrink=0.35,
        hard_floor=0.55,
        flip_young_p_eff_floor=0.55,
        tcn_ref=TradeDirection.PUT,
        tape="CALL",
    )
    assert young is False
    assert discord is True
    assert p_eff == 0.10
    assert floor == 0.55


def test_loss_clf_bootstrap_blocks_flip(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.64, n_train=64, auto_learn=False, bootstrap=True)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "bootstrap"
    assert metrics["loss_clf_bootstrap"] is True


def test_loss_clf_no_auto_learn_blocks_flip(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.64, n_train=64, auto_learn=False, bootstrap=False)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_flip_blocked"] == "no_auto_learn"


def test_loss_clf_flip_above_floor_mature(monkeypatch):
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "calibrated_prob": 0.6,
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.56, n_train=32)
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=_orch()) is False
    assert metrics["exec_direction"] == "PUT"
    assert metrics["resolved_direction"] == "PUT"
    assert metrics["loss_clf_flip"] is True
    assert metrics["loss_clf_p_eff"] == 0.56
    assert metrics["loss_clf_flip_floor"] == 0.55
    assert metrics.get("loss_clf_flip_blocked") is None
    assert metrics["execution_candidate_ready"] is True


def test_loss_clf_ok_below_floor_mature(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "calibrated_prob": 0.4,
        "execution_candidate_ready": True,
        "scale_tape_consensus": "PUT",
    }
    _patch_predict(monkeypatch, p_loss=0.21, n_train=32)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics["exec_direction"] == "PUT"
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["loss_clf_p_eff"] == 0.21
    assert metrics["loss_clf_flip_floor"] == 0.55


def test_loss_clf_young_low_p_no_flip(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "scale_tape_consensus": "CALL",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.10, n_train=12)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_young_shrink"] is True
    assert metrics["loss_clf_flip_floor"] == 0.55
    assert metrics["loss_clf_p_eff"] < 0.55


def test_loss_clf_young_high_p_flips(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "scale_tape_consensus": "PUT",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.71, n_train=12)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics["loss_clf_flip"] is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics["loss_clf_p_eff"] >= 0.55
    assert metrics["loss_clf_flip_floor"] == 0.55


def test_loss_clf_young_mid_p_no_flip(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "scale_tape_consensus": "PUT",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.577, n_train=27)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_p_eff"] < 0.55


def test_loss_clf_tape_discord_does_not_force_flip_mature(monkeypatch):
    metrics = {
        "tcn_direction": "PUT",
        "exec_direction": "PUT",
        "scale_tape_consensus": "CALL",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.10, n_train=64)
    assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=_orch()) is False
    assert metrics.get("loss_clf_flip") is not True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["loss_clf_tape_discord"] is True
    assert metrics["loss_clf_p_eff"] == 0.10


def test_loss_clf_force_noop():
    metrics = {"execution_candidate_ready": True, "exec_direction": "CALL"}
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=_orch(), force=True) is False
    assert metrics["exec_direction"] == "CALL"


def test_only_loss_classifier_gate_assigns_loss_clf_flip():
    gate_path = Path(__file__).resolve().parents[3] / "src" / "application" / "services" / "loss_classifier_gate.py"
    src = gate_path.read_text(encoding="utf-8")
    assert 'metrics["loss_clf_flip"] = True' in src
    assert "flip_floor" in src
    assert "bootstrap" in src
    app_src = Path(__file__).resolve().parents[3] / "src"
    offenders = []
    for path in app_src.rglob("*.py"):
        if path.name == "loss_classifier_gate.py":
            continue
        text = path.read_text(encoding="utf-8")
        if 'loss_clf_flip"] = True' in text or "loss_clf_flip'] = True" in text:
            offenders.append(str(path.relative_to(app_src)))
    assert offenders == []


def test_stamp_flip_ctx_persists_on_gate(monkeypatch):
    orch = SimpleNamespace(
        config={"infra": {"loss_classifier": {"enabled": True}}},
        _active_cycle_id=1,
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 0.0, bankroll=1000.0),
        state=SimpleNamespace(balance=1000.0),
    )
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "execution_candidate_ready": True,
    }
    _patch_predict(monkeypatch, p_loss=0.56, n_train=32)
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="1HZ75V") is False
    ctx = orch._loss_clf_flip_ctx["1HZ75V"]
    assert ctx["flip"] is True
    assert ctx["p_eff"] == pytest.approx(0.56)
