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

META_EDGE_CLIP_LOW = -1.0
META_EDGE_CLIP_HIGH = 0.85
META_FIT_MIN_N = 2


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
    n_samples = len(buffer_y)
    min_child = max(8, min(16, max(2, n_samples // 4)))
    model = lgb.LGBMRegressor(
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=6,
        max_depth=4,
        max_bin=63,
        min_child_samples=min_child,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.35,
        objective="huber",
        extra_trees=True,
        n_jobs=2,
        verbosity=-1,
        random_state=42,
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


def meta_retrain_floor(retrain_min_n: int) -> int:
    return max(META_FIT_MIN_N, int(retrain_min_n))


def should_retrain_meta(*, buffer_n: int, retrain_min_n: int) -> bool:
    return int(buffer_n) >= meta_retrain_floor(retrain_min_n)


def bundle_n_train(bundle: dict[str, Any]) -> int:
    try:
        return max(0, int(bundle.get("n_train") or 0))
    except (TypeError, ValueError):
        return 0


def is_online_bundle(path: Path, bundle: dict[str, Any]) -> bool:
    name = str(path.name)
    if name.startswith("meta_online_"):
        return True
    return bool(bundle.get("auto_learn_applied"))


def is_tiny_online_bundle(path: Path, bundle: dict[str, Any], *, floor: int) -> bool:
    return is_online_bundle(path, bundle) and bundle_n_train(bundle) < int(floor)


def rank_meta_bundle(path: Path, bundle: dict[str, Any]) -> tuple[int, int, float]:
    n_train = bundle_n_train(bundle)
    offline_bonus = 0 if is_online_bundle(path, bundle) else 1
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (n_train, offline_bonus, mtime)


def clamp_meta_edge(edge: float, *, auto_learn: bool, n_train: int, floor: int) -> float:
    value = float(edge)
    if bool(auto_learn) and int(n_train) < int(floor):
        return max(META_EDGE_CLIP_LOW, min(META_EDGE_CLIP_HIGH, value))
    return value
