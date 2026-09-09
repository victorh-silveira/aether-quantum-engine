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
    cal_temperature: float = 1.0,
    cal_ece: float = 1.0,
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
            "cal_temperature": float(cal_temperature),
            "cal_ece": float(cal_ece),
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
    mature = n_samples >= 32
    min_child = max(8, min(20, n_samples // 8)) if mature else max(2, min(8, n_samples // 4))
    model = lgb.LGBMClassifier(
        n_estimators=80 if mature else 50,
        learning_rate=0.05,
        num_leaves=12 if mature else 10,
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


def fit_seed_classifier(buffer_x: list[list[float]], buffer_y: list[int]) -> Any:
    n_samples = len(buffer_y)
    min_child = max(4, min(12, n_samples // 3))
    model = lgb.LGBMClassifier(
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=4,
        max_depth=3,
        max_bin=63,
        min_child_samples=min_child,
        class_weight="balanced",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.4,
        extra_trees=True,
        n_jobs=2,
        verbosity=-1,
        random_state=42,
    )
    model.fit(np.asarray(buffer_x, dtype=np.float64), np.asarray(buffer_y, dtype=np.int32))
    return model


def predict_p_loss(model: Any, vector: list[float], *, temperature: float = 1.0) -> float:
    proba = np.asarray(model.predict_proba(np.asarray([vector], dtype=np.float64))[0], dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    loss_idx = list(classes).index(1) if 1 in classes else len(proba) - 1
    temp = float(temperature)
    if temp <= 1.0 + 1e-12 or proba.size < 2:
        return float(proba[loss_idx])
    clipped = np.clip(proba, 1e-9, 1.0 - 1e-9)
    logits = np.log(clipped) / temp
    logits = logits - float(np.max(logits))
    expv = np.exp(logits)
    soft = expv / float(np.sum(expv))
    return float(soft[loss_idx])


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


def live_like_synthetic_xy(
    feature_dim: int,
    *,
    n: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X/y na escala do vetor live 24D (evita OOD N(0,1) com p_loss saturado)."""
    rng = np.random.default_rng(int(seed))
    rows = int(n)
    dim = int(feature_dim)
    x_arr = np.zeros((rows, dim), dtype=np.float64)
    x_arr[:, 0] = rng.uniform(0.0, 0.15, size=rows)
    if dim > 1:
        x_arr[:, 1] = rng.uniform(0.42, 0.58, size=rows)
    if dim > 2:
        x_arr[:, 2] = rng.uniform(0.0, 0.12, size=rows)
    for idx in (3, 14, 15, 23):
        if idx < dim:
            x_arr[:, idx] = rng.integers(0, 2, size=rows).astype(np.float64)
    if dim > 6:
        regime = rng.integers(0, 3, size=rows)
        x_arr[:, 4] = (regime == 0).astype(np.float64)
        x_arr[:, 5] = (regime == 1).astype(np.float64)
        x_arr[:, 6] = (regime == 2).astype(np.float64)
    if dim > 7:
        x_arr[:, 7] = rng.integers(0, 2, size=rows).astype(np.float64)
    if dim > 8:
        x_arr[:, 8] = rng.uniform(0.0, 0.6, size=rows)
    if dim > 9:
        x_arr[:, 9] = rng.uniform(0.0, 0.4, size=rows)
    if dim > 10:
        x_arr[:, 10] = np.clip(rng.normal(-0.02, 0.06, size=rows), -3.0, 3.0)
    if dim > 11:
        x_arr[:, 11] = rng.uniform(0.45, 0.70, size=rows)
    if dim > 12:
        x_arr[:, 12] = rng.uniform(0.0, 0.5, size=rows)
    if dim > 13:
        x_arr[:, 13] = rng.uniform(0.35, 0.70, size=rows)
    if dim > 16:
        x_arr[:, 16] = rng.uniform(0.05, 0.55, size=rows)
    if dim > 17:
        x_arr[:, 17] = np.clip(rng.normal(1.0, 0.35, size=rows), -3.0, 3.0)
    if dim > 18:
        x_arr[:, 18] = rng.uniform(0.42, 0.58, size=rows)
    if dim > 19:
        x_arr[:, 19] = np.clip(rng.normal(0.0, 0.5, size=rows), -3.0, 3.0)
    if dim > 20:
        x_arr[:, 20] = rng.uniform(0.50, 0.65, size=rows)
    if dim > 21:
        x_arr[:, 21] = np.clip(rng.normal(0.0, 0.35, size=rows), -3.0, 3.0)
    if dim > 22:
        x_arr[:, 22] = np.clip(rng.normal(0.0, 0.8, size=rows), -3.0, 3.0)
    risk = (
        0.6 * (x_arr[:, 3] if dim > 3 else 0.0)
        + 0.5 * (x_arr[:, 7] if dim > 7 else 0.0)
        + 0.4 * (x_arr[:, 9] if dim > 9 else 0.0)
        - 2.5 * (x_arr[:, 10] if dim > 10 else 0.0)
        - 1.0 * x_arr[:, 0]
        + rng.normal(0.0, 0.25, size=rows)
    )
    order = np.argsort(risk)
    y_arr = np.zeros(rows, dtype=np.int32)
    y_arr[order[rows // 2 :]] = 1
    if len(np.unique(y_arr)) < 2:
        y_arr[0] = 0
        y_arr[1] = 1
    return x_arr, y_arr


def is_stale_gaussian_bootstrap(version: str) -> bool:
    """True se bundle ainda e o seed OOD legado loss_bootstrap_synth."""
    ver = str(version or "").strip()
    return ver == "loss_bootstrap_synth" or ver.startswith("loss_bootstrap_synth")


def seed_bootstrap_classifier(
    models_dir: Path,
    feature_dim: int,
    feature_names: tuple[str, ...],
    *,
    n: int = 64,
    seed: int = 42,
) -> tuple[dict[str, Any], Path] | None:
    """Treina LGBM live-like e persiste seed pronto (veto_ready se n>=READY_N)."""
    try:
        x_arr, y_arr = live_like_synthetic_xy(int(feature_dim), n=int(n), seed=int(seed))
        model = fit_seed_classifier(x_arr.tolist(), y_arr.tolist())
        models_dir.mkdir(parents=True, exist_ok=True)
        version = "loss_bootstrap_live64"
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
