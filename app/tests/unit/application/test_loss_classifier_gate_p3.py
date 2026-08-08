"""Testes do loss-classifier (parte 3): policy, runtime, buffer e flip helpers."""

from __future__ import annotations

import pytest

from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.domain.models.trade import TradeDirection


def test_should_retrain_after_learn_loss_forces_when_ready():
    import importlib.util
    from pathlib import Path

    policy_path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "loss-classifier" / "learn_policy.py"
    spec = importlib.util.spec_from_file_location("loss_learn_policy", policy_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=10,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=8,
            buffer_loss=2,
        )
        is True
    )
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=2,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=0,
            buffer_loss=2,
        )
        is False
    )
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=20,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=7,
            buffer_loss=13,
        )
        is False
    )
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=1,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=8,
            buffer_loss=0,
        )
        is False
    )
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=2,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=1,
            buffer_loss=1,
            bootstrap_active=True,
        )
        is True
    )
    assert (
        mod.should_retrain_after_learn(
            label="LOSS",
            buffer_n=2,
            retrain_min_n=24,
            retrain_on_loss_min_n=2,
            buffer_win=0,
            buffer_loss=2,
            bootstrap_active=True,
        )
        is False
    )
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=2, retrain_min_n=24, retrain_on_loss_min_n=2) is False
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=24, retrain_min_n=24, retrain_on_loss_min_n=2) is True
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=25, retrain_min_n=24, retrain_on_loss_min_n=2) is False
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=32, retrain_min_n=24, retrain_on_loss_min_n=2) is True
    assert mod.retrain_min_for_label(label="LOSS", retrain_min_n=24, retrain_on_loss_min_n=2) == 2
    assert mod.retrain_min_for_label(label="WIN", retrain_min_n=24, retrain_on_loss_min_n=2) == 24


def test_fit_classifier_accepts_imbalanced_classes():
    import importlib.util
    from pathlib import Path

    pytest.importorskip("lightgbm")
    runtime_path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "loss-classifier" / "runtime.py"
    spec = importlib.util.spec_from_file_location("loss_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buffer_x = [[float(i % 3)] * 24 for i in range(30)]
    buffer_y = [1] * 25 + [0] * 5
    model = mod.fit_classifier(buffer_x, buffer_y)
    assert hasattr(model, "predict_proba")
    params = model.get_params()
    assert int(params.get("min_child_samples", 0)) == 15
    assert params.get("class_weight") == "balanced"
    p_loss = mod.predict_p_loss(model, [0.0] * 24)
    assert 0.0 <= p_loss <= 1.0
    assert mod.is_bootstrap_bundle({"bootstrap": True, "model_version": "x"}) is True
    assert mod.is_bootstrap_bundle({"bootstrap": False, "model_version": "loss_bootstrap_synth"}) is True
    assert mod.is_bootstrap_bundle({"bootstrap": False, "model_version": "loss_1_n40"}) is False


def test_learn_buffer_io_roundtrip(tmp_path):
    import importlib.util
    from pathlib import Path

    io_path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "loss-classifier" / "buffer_io.py"
    spec = importlib.util.spec_from_file_location("loss_buffer_io", io_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x_rows = [[0.1] * 24, [0.2] * 24]
    y_rows = [0, 1]
    mod.save_learn_buffer(tmp_path, x_rows, y_rows)
    loaded = mod.load_learn_buffer(tmp_path)
    assert loaded is not None
    assert loaded[0] == x_rows
    assert loaded[1] == y_rows
    assert mod.buffer_class_counts(y_rows) == {"win": 1, "loss": 1, "n": 2}


def test_loss_classifier_flip_helpers():
    from src.application.services.loss_classifier_flip import (
        apply_loss_flip,
        is_seed_model,
        resolve_soft_kelly_mult,
        scale_confirms_ref,
    )

    assert is_seed_model({"auto_learn_applied": False}, require_auto_learn=False) is False
    assert is_seed_model({"auto_learn_applied": False}, require_auto_learn=True) is True
    metrics = {"scale_vote_call_n": "x", "scale_vote_put_n": 1}
    assert scale_confirms_ref(metrics, TradeDirection.CALL) is False
    put_votes = {"scale_vote_call_n": 0, "scale_vote_put_n": 5}
    assert scale_confirms_ref(put_votes, TradeDirection.PUT) is True
    cfg = {
        "veto_p_loss_floor": 0.50,
        "soft_p_loss_high": 0.85,
        "soft_kelly_mult": 0.55,
        "soft_kelly_mult_high": 0.20,
        "hard_p_loss_floor": 0.90,
        "soft_max_stake_pct_high": 0.002,
    }
    flipped = apply_loss_flip({}, TradeDirection.CALL, cfg=cfg)
    assert flipped == TradeDirection.PUT
    assert resolve_soft_kelly_mult(0.50, cfg) == pytest.approx(0.55)


def test_build_loss_feature_invalid_flow_and_edge():
    vector = build_loss_feature_vector(
        {
            "flow_features": {"micro_tick_acceleration": "bad"},
            "predicted_payoff_edge": object(),
            "edge_zscore": 0.12,
        },
        TradeDirection.CALL,
    )
    assert vector[19] == 0.0
    assert vector[10] == pytest.approx(0.12)
