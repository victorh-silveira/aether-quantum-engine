from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np


logger = logging.getLogger("META")


def load_learn_buffer(path: Path) -> tuple[list[list[float]], list[float]]:
    if not path.is_file():
        return [], []
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        logger.warning("Falha ao ler buffer meta: %s", exc)
        return [], []
    if not isinstance(payload, dict):
        return [], []
    xs = payload.get("x")
    ys = payload.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list):
        return [], []
    return [list(row) for row in xs], [float(v) for v in ys]


def save_learn_buffer(path: Path, xs: list[list[float]], ys: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"x": xs, "y": ys}, handle)


def fit_regressor(buffer_x: list[list[float]], buffer_y: list[float]) -> Any:
    model = lgb.LGBMRegressor(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1,
    )
    model.fit(np.asarray(buffer_x, dtype=np.float64), np.asarray(buffer_y, dtype=np.float64))
    return model


def persist_regressor_bundle(
    models_dir: Path,
    model: Any,
    *,
    n_train: int,
    feature_names: list[str],
    feature_dim: int,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    version = f"meta_online_{int(time.time())}_n{n_train}"
    path = models_dir / f"{version}.pkl"
    joblib.dump(
        {
            "model": model,
            "model_type": "regressor",
            "feature_names": list(feature_names),
            "n_train": int(n_train),
            "auto_learn_applied": True,
            "model_version": version,
            "feature_dim": int(feature_dim),
        },
        path,
    )
    return path


def should_retrain_meta(*, buffer_n: int, retrain_min_n: int) -> bool:
    return int(buffer_n) >= max(1, int(retrain_min_n))
