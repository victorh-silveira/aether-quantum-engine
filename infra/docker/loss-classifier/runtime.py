from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np


logger = logging.getLogger("LOSS_CLF")


def is_bootstrap_bundle(bundle: dict[str, Any], *, version: str = "") -> bool:
    if bool(bundle.get("bootstrap")):
        return True
    ver = str(version or bundle.get("model_version") or "")
    return ver.startswith("loss_bootstrap")


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
            "bootstrap": False,
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
    n_samples = len(buffer_y)
    min_child = max(2, min(8, n_samples // 4))
    model = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=10,
        max_bin=63,
        min_child_samples=min_child,
        class_weight="balanced",
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.15,
        extra_trees=True,
        n_jobs=2,
        verbosity=-1,
        random_state=42,
    )
    model.fit(np.asarray(buffer_x, dtype=np.float64), np.asarray(buffer_y, dtype=np.int32))
    return model


def predict_p_loss(model: Any, vector: list[float]) -> float:
    proba = model.predict_proba(np.asarray([vector], dtype=np.float64))[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return float(proba[list(classes).index(1)])
    return float(proba[-1])


def is_collapsed_classifier(
    model: Any,
    buffer_x: list[list[float]],
    *,
    min_std: float = 0.02,
    min_range: float = 0.05,
) -> bool:
    if not buffer_x:
        return True
    probs = np.asarray([predict_p_loss(model, row) for row in buffer_x], dtype=np.float64)
    if probs.size < 2:
        return True
    spread = float(probs.max() - probs.min())
    return float(probs.std()) < float(min_std) or spread < float(min_range)


def seed_bootstrap_classifier(
    models_dir: Path,
    feature_dim: int,
    feature_names: tuple[str, ...],
    *,
    n: int = 64,
    seed: int = 42,
) -> tuple[dict[str, Any], Path] | None:
    """Treina LGBM sintetico e persiste seed pronto (veto_ready se n>=READY_N)."""
    try:
        rng = np.random.default_rng(int(seed))
        x_arr = rng.normal(size=(int(n), int(feature_dim)))
        y_arr = (x_arr[:, 0] + 0.3 * x_arr[:, min(7, feature_dim - 1)] > 0.0).astype(np.int32)
        if len(np.unique(y_arr)) < 2:
            y_arr[0] = 0
            y_arr[1] = 1
        model = fit_classifier(x_arr.tolist(), y_arr.tolist())
        models_dir.mkdir(parents=True, exist_ok=True)
        version = "loss_bootstrap_synth"
        path = models_dir / f"{version}.pkl"
        bundle = {
            "model": model,
            "model_type": "classifier",
            "feature_names": list(feature_names),
            "n_train": int(len(y_arr)),
            "auto_learn_applied": False,
            "bootstrap": True,
            "model_version": version,
            "feature_dim": int(feature_dim),
        }
        joblib.dump(bundle, path)
        return bundle, path
    except Exception as exc:
        logger.warning("seed bootstrap falhou: %s", exc)
        return None
