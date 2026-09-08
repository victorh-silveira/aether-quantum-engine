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
