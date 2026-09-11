"""Calibracao meta/loss: piso retrain, clamp edge, temperatura, FLIP no piso."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    import numpy as np
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
    mod.drop_sklearn_feature_names(model)
    assert "feature_names_in_" not in model.__dict__
    dummy = type("M", (), {})()
    assert mod.drop_sklearn_feature_names(dummy) is dummy
    p1 = mod.predict_p_loss(model, row, temperature=1.0)
    p2 = mod.predict_p_loss(model, row, temperature=2.0)
    assert abs(p2 - 0.5) <= abs(p1 - 0.5) + 1e-9
    assert 0.0 <= p2 <= 1.0
    one = mod.predict_proba_numpy(model, row)
    assert one.shape[0] == 1

    class _FakeProba:
        def predict_proba(self, _x):
            return np.asarray([[0.25, 0.75]])

    assert float(mod.predict_proba_numpy(_FakeProba(), [[0.0] * 24])[0, 1]) == 0.75

    class _FlipBooster:
        def predict(self, _x):
            return np.asarray([0.8])

    class _FlipModel:
        booster_ = _FlipBooster()
        classes_ = [1, 0]

    flipped = mod.predict_proba_numpy(_FlipModel(), [0.0])
    assert abs(float(flipped[0, 0]) - 0.8) < 1e-12

    class _MultiBooster:
        def predict(self, _x):
            return np.asarray([[0.1, 0.9]])

    class _MultiModel:
        booster_ = _MultiBooster()

    multi = mod.predict_proba_numpy(_MultiModel(), [[0.0]])
    assert abs(float(multi[0, 1]) - 0.9) < 1e-12

    class _Slotted:
        __slots__ = ()

    assert mod.drop_sklearn_feature_names(_Slotted()) is not None


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
    assert mod.clamp_meta_edge(2.5, auto_learn=True, n_train=40, floor=32) == 0.85
    assert mod.apply_label_scale(0.4, {"label_scale": 2.0}) == pytest.approx(0.8)
    assert mod.apply_label_scale(0.4, {}) == pytest.approx(0.4)
    assert mod.apply_label_scale(0.4, {"label_scale": 0.0}) == pytest.approx(0.4)
    assert mod.apply_label_scale(0.4, {"label_scale": "x"}) == pytest.approx(0.4)
    assert mod.bundle_val_mae({"val_mae": "x"}) is None
    assert mod.buffer_abs_mae(None, [], []) is None
    online = Path("meta_online_1_n5.pkl")
    offline = Path("meta_lgbm.pkl")
    assert mod.is_tiny_online_bundle(online, {"auto_learn_applied": True, "n_train": 5}, floor=32)
    assert not mod.is_tiny_online_bundle(offline, {"auto_learn_applied": False, "n_train": 200}, floor=32)
    r_on = mod.rank_meta_bundle(online, {"auto_learn_applied": True, "n_train": 5})
    r_off = mod.rank_meta_bundle(offline, {"auto_learn_applied": False, "n_train": 200})
    assert r_off > r_on
    kept = mod.select_meta_bundle(
        [(online, {"auto_learn_applied": True, "n_train": 40}), (offline, {"n_train": 200, "val_mae": 0.1})],
        floor=32,
        buffer_mae=0.5,
    )
    assert kept is not None
    assert kept[0] == offline
    promoted = mod.select_meta_bundle(
        [(online, {"auto_learn_applied": True, "n_train": 40}), (offline, {"n_train": 200, "val_mae": 0.1})],
        floor=32,
        buffer_mae=0.15,
    )
    assert promoted is not None
    assert promoted[0] == online
    assert mod.online_may_replace_offline(
        {"n_train": 40},
        {"val_mae": 0.1},
        buffer_mae=0.15,
        floor=32,
    )


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


def test_loss_bootstrap_exit_is_four_not_ready_n():
    import sys

    sidecar = str(_repo_root() / "infra" / "docker" / "loss-classifier")
    if sidecar not in sys.path:
        sys.path.insert(0, sidecar)
    import learn_policy as policy

    assert policy.bootstrap_retrain_floor(retrain_on_loss_min_n=4, bootstrap_exit_n=4) == 4
    kw = {
        "label": "LOSS",
        "retrain_min_n": 12,
        "retrain_on_loss_min_n": 4,
        "buffer_win": 8,
        "buffer_loss": 3,
        "bootstrap_active": True,
        "bootstrap_exit_n": 4,
    }
    assert policy.should_retrain_after_learn(buffer_n=3, **kw) is False
    assert policy.should_retrain_after_learn(buffer_n=4, **kw) is True
    assert policy.retrain_skipped_reason(buffer_n=3, **kw) == "bootstrap_wait:3/4"
    keep_3 = policy.bootstrap_seed_keep_detail(
        bootstrap=True,
        buffer_n=3,
        n_classes=2,
        exit_n=4,
        floor=4,
    )
    keep_4 = policy.bootstrap_seed_keep_detail(
        bootstrap=True,
        buffer_n=4,
        n_classes=2,
        exit_n=4,
        floor=4,
    )
    assert keep_3 == "seed_keep n<4"
    assert keep_4 is None
    assert keep_4 != "seed_keep n<32"
    src = (_repo_root() / "infra" / "docker" / "loss-classifier" / "app.py").read_text(encoding="utf-8")
    assert "seed_keep n<{int(READY_N)}" not in src
    assert "bootstrap_seed_keep_detail" in src
