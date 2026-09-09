"""Calibracao meta/loss: piso retrain, clamp edge, temperatura, FLIP no piso."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.domain.models.trade import TradeDirection


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "infra" / "docker").is_dir():
            return parent
    return Path.cwd()


def test_flip_above_floor_inverts_and_keeps_candidate():
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = 1000.0
    orch._log_dedupe = {}
    orch._active_cycle_id = 77
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "tcn_direction": "CALL",
        "kelly_fraction_scale": 1.0,
        "calibrated_prob": 0.55,
    }
    with (
        patch(
            "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
            return_value={
                "p_loss": 0.92,
                "veto": True,
                "auto_learn_applied": True,
                "model_version": "loss_live",
                "n_train": 64,
                "veto_ready": True,
                "bootstrap": False,
            },
        ),
        patch(
            "src.application.services.loss_classifier_gate.build_loss_feature_vector",
            return_value=[0.0] * 24,
        ),
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=orch, symbol="1HZ75V") is False
    assert metrics.get("exec_direction") == "PUT"
    assert metrics.get("loss_clf_flip") is True
    assert metrics.get("gate_reason") != "loss_clf"
    assert metrics.get("execution_candidate_ready") is True


def test_predict_p_loss_temperature_softens():
    from lightgbm import LGBMClassifier

    from scripts.operations.train_loss_classifier import live_like_synthetic_xy

    x_arr, y_arr = live_like_synthetic_xy(24, n=64, seed=42)
    model = LGBMClassifier(
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=4,
        max_depth=3,
        min_child_samples=8,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )
    model.fit(x_arr, y_arr)
    row = x_arr[int(y_arr.argmax())].tolist()
    runtime_path = _repo_root() / "infra" / "docker" / "loss-classifier" / "runtime.py"
    spec = importlib.util.spec_from_file_location("loss_runtime_temp", runtime_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p1 = mod.predict_p_loss(model, row, temperature=1.0)
    p2 = mod.predict_p_loss(model, row, temperature=2.0)
    assert abs(p2 - 0.5) <= abs(p1 - 0.5) + 1e-9
    assert 0.0 <= p2 <= 1.0


def test_meta_clamp_and_tiny_online_rank():
    path = _repo_root() / "infra" / "docker" / "meta-classifier" / "learn_runtime.py"
    spec = importlib.util.spec_from_file_location("meta_learn_runtime", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.meta_retrain_floor(32) == 32
    assert mod.should_retrain_meta(buffer_n=5, retrain_min_n=32) is False
    assert mod.should_retrain_meta(buffer_n=32, retrain_min_n=32) is True
    assert mod.clamp_meta_edge(2.5, auto_learn=True, n_train=5, floor=32) == 0.85
    assert mod.clamp_meta_edge(-3.0, auto_learn=True, n_train=5, floor=32) == -1.0
    assert mod.clamp_meta_edge(2.5, auto_learn=True, n_train=40, floor=32) == 2.5
    online = Path("meta_online_1_n5.pkl")
    offline = Path("meta_lgbm.pkl")
    assert mod.is_tiny_online_bundle(online, {"auto_learn_applied": True, "n_train": 5}, floor=32)
    assert not mod.is_tiny_online_bundle(offline, {"auto_learn_applied": False, "n_train": 200}, floor=32)
    r_on = mod.rank_meta_bundle(online, {"auto_learn_applied": True, "n_train": 5})
    r_off = mod.rank_meta_bundle(offline, {"auto_learn_applied": False, "n_train": 200})
    assert r_off > r_on


def test_loss_sidecar_calib_nll_ece_and_young_temp():
    import sys

    sidecar = str(_repo_root() / "infra" / "docker" / "loss-classifier")
    if sidecar not in sys.path:
        sys.path.insert(0, sidecar)
    import calib as loss_calib

    assert loss_calib.binary_nll([], []) == float("inf")
    assert loss_calib.binary_nll([0.2], [0, 1]) == float("inf")
    assert loss_calib.binary_ece([0.2], [1]) == 1.0
    nll = loss_calib.binary_nll([0.2, 0.8], [0, 1])
    assert nll < 1.0
    ece = loss_calib.binary_ece([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1])
    assert ece < 0.2
    dummy = type("M", (), {})()
    assert loss_calib.fit_temperature(dummy, [[0.0] * 24] * 8, [0, 1] * 4) == 1.0


def test_loss_sidecar_fit_temperature_picks_grid(monkeypatch):
    import sys

    sidecar = str(_repo_root() / "infra" / "docker" / "loss-classifier")
    if sidecar not in sys.path:
        sys.path.insert(0, sidecar)
    import calib as loss_calib

    rows = [[0.0] * 24 for _ in range(32)]
    labels = [0] * 16 + [1] * 16

    def _fake_predict(_model, _row, temperature=1.0):
        return 0.9 if float(temperature) >= 1.5 else 0.51

    monkeypatch.setattr(loss_calib, "predict_p_loss", _fake_predict)
    chosen = loss_calib.fit_temperature(object(), rows, labels)
    assert chosen == 0.70
