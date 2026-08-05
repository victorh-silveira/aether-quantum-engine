from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np


logger = logging.getLogger("LOSS_CLF")


def persist_bundle(
    models_dir: Path,
    model: Any,
    n_train: int,
    feature_names: tuple[str, ...],
    feature_dim: int,
    *,
    auto_learn: bool,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    version = f"loss_{int(time.time())}_n{n_train}"
    path = models_dir / f"{version}.pkl"
    joblib.dump(
        {
            "model": model,
            "model_type": "classifier",
            "feature_names": list(feature_names),
            "n_train": int(n_train),
            "auto_learn_applied": bool(auto_learn),
            "model_version": version,
            "feature_dim": int(feature_dim),
        },
        path,
    )
    return path


def load_latest_classifier(models_dir: Path) -> tuple[dict[str, Any], Path] | None:
    if not models_dir.is_dir():
        return None
    candidates = sorted(models_dir.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            logger.warning("Falha ao carregar %s: %s", path, exc)
            continue
        if not isinstance(bundle, dict) or bundle.get("model") is None:
            continue
        if not callable(getattr(bundle["model"], "predict_proba", None)):
            continue
        return bundle, path
    return None


def fit_classifier(buffer_x: list[list[float]], buffer_y: list[int]) -> Any:
    model = lgb.LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1,
    )
    model.fit(np.asarray(buffer_x, dtype=np.float64), np.asarray(buffer_y, dtype=np.int32))
    return model


def predict_p_loss(model: Any, vector: list[float]) -> float:
    proba = model.predict_proba(np.asarray([vector], dtype=np.float64))[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return float(proba[list(classes).index(1)])
    return float(proba[-1])
