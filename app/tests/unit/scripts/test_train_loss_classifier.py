"""Bootstrap live-like do loss-classifier (anti OOD N(0,1))."""

from __future__ import annotations

import numpy as np

from scripts.operations.train_loss_classifier import FEATURE_DIM, live_like_synthetic_xy


def test_live_like_synthetic_xy_scales_and_balance():
    x_arr, y_arr = live_like_synthetic_xy(FEATURE_DIM, n=64, seed=42)
    assert x_arr.shape == (64, FEATURE_DIM)
    assert set(np.unique(y_arr).tolist()) == {0, 1}
    assert float(x_arr[:, 0].min()) >= 0.0
    assert float(x_arr[:, 0].max()) <= 0.15 + 1e-9
    assert float(x_arr[:, 1].min()) >= 0.42 - 1e-9
    assert float(x_arr[:, 1].max()) <= 0.58 + 1e-9
    win_frac = float(np.mean(y_arr == 0))
    assert 0.40 <= win_frac <= 0.60


def test_seed_model_p_loss_not_saturated_on_typical_live_vector():
    import lightgbm as lgb

    x_arr, y_arr = live_like_synthetic_xy(FEATURE_DIM, n=64, seed=42)
    model = lgb.LGBMClassifier(
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=4,
        max_depth=3,
        min_child_samples=8,
        class_weight="balanced",
        reg_alpha=0.2,
        reg_lambda=0.4,
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )
    model.fit(x_arr, y_arr)
    classes = list(model.classes_)
    loss_idx = classes.index(1) if 1 in classes else -1
    train_probs = model.predict_proba(x_arr)[:, loss_idx]
    assert float(np.median(train_probs)) <= 0.70
    assert float(np.mean(train_probs > 0.90)) < 0.25
    typical = np.zeros((1, FEATURE_DIM), dtype=np.float64)
    typical[0, 0] = 0.03
    typical[0, 1] = 0.53
    typical[0, 6] = 1.0
    typical[0, 10] = -0.02
    typical[0, 11] = 0.55
    typical[0, 18] = 0.53
    typical[0, 20] = 0.59
    p_loss = float(model.predict_proba(typical)[0, loss_idx])
    assert p_loss < 0.85


def test_load_real_settle_xy_rejects_short_or_collapsed(tmp_path):
    import joblib

    from scripts.operations.train_loss_classifier import load_real_settle_xy

    assert load_real_settle_xy(tmp_path) is None
    joblib.dump({"x": [[0.0] * FEATURE_DIM] * 4, "y": [0, 1, 0, 1]}, tmp_path / "learn_buffer.pkl")
    assert load_real_settle_xy(tmp_path) is None
    joblib.dump(
        {"x": [[0.0] * FEATURE_DIM] * 12, "y": [0] * 12},
        tmp_path / "learn_buffer.pkl",
    )
    assert load_real_settle_xy(tmp_path) is None
    joblib.dump("bad", tmp_path / "learn_buffer.pkl")
    assert load_real_settle_xy(tmp_path) is None


def test_load_real_settle_xy_accepts_mixed_classes(tmp_path):
    import joblib

    from scripts.operations.train_loss_classifier import load_real_settle_xy

    rows = [[float(i % 3) * 0.01] + [0.0] * (FEATURE_DIM - 1) for i in range(12)]
    labels = [0] * 6 + [1] * 6
    joblib.dump({"x": rows, "y": labels}, tmp_path / "learn_buffer.pkl")
    loaded = load_real_settle_xy(tmp_path)
    assert loaded is not None
    x_arr, y_arr = loaded
    assert x_arr.shape == (12, FEATURE_DIM)
    assert set(y_arr.tolist()) == {0, 1}


def test_main_writes_real_seed_when_buffer_ready(tmp_path, monkeypatch):
    import joblib

    from scripts.operations.train_loss_classifier import main

    rows = [[0.02] + [0.0] * (FEATURE_DIM - 1) for _ in range(12)]
    labels = [0] * 6 + [1] * 6
    joblib.dump({"x": rows, "y": labels}, tmp_path / "learn_buffer.pkl")
    monkeypatch.setattr("sys.argv", ["train_loss_classifier.py", "--out-dir", str(tmp_path)])
    assert main() == 0
    seed = tmp_path / "loss_seed_real12.pkl"
    assert seed.is_file()
    payload = joblib.load(seed)
    assert payload["bootstrap"] is False
    assert payload["auto_learn_applied"] is True
    assert payload["n_train"] == 12
