from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

_ML_ROOT = Path(__file__).resolve().parent.parent
if (_ML_ROOT / "ml_common" / "__init__.py").is_file() and str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml_common.persist import atomic_joblib_dump, atomic_pickle_dump
from ml_common.schema import bundle_schema_hash_ok

logger = logging.getLogger("META")

META_EDGE_CLIP_LOW = -1.0
META_EDGE_CLIP_HIGH = 0.85
META_FIT_MIN_N = 2
META_ONLINE_GATE_N = 32
META_ONLINE_MAE_MULT = 2.0


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
    atomic_pickle_dump({"x": xs, "y": ys}, path)


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
    schema_hash: str = "",
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    version = f"meta_online_{int(time.time())}_n{n_train}"
    path = models_dir / f"{version}.pkl"
    atomic_joblib_dump(
        {
            "model": model,
            "model_type": "regressor",
            "feature_names": list(feature_names),
            "n_train": int(n_train),
            "auto_learn_applied": True,
            "model_version": version,
            "feature_dim": int(feature_dim),
            "schema_hash": str(schema_hash),
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
    offline_bonus = 0 if is_online_bundle(path, bundle) else 2
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (offline_bonus, n_train, mtime)


def bundle_val_mae(bundle: dict[str, Any]) -> float | None:
    raw = bundle.get("val_mae")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0.0:
        return None
    return value


def buffer_abs_mae(model: Any, xs: list[list[float]], ys: list[float]) -> float | None:
    if not xs or len(xs) != len(ys):
        return None
    pred = np.asarray(model.predict(np.asarray(xs, dtype=np.float64)), dtype=np.float64).reshape(-1)
    target = np.asarray(ys, dtype=np.float64).reshape(-1)
    if pred.size != target.size or pred.size == 0:
        return None
    return float(np.mean(np.abs(pred - target)))


def online_may_replace_offline(
    online_bundle: dict[str, Any],
    offline_bundle: dict[str, Any],
    *,
    buffer_mae: float | None,
    floor: int = META_ONLINE_GATE_N,
) -> bool:
    if bundle_n_train(online_bundle) < int(floor):
        return False
    val_mae = bundle_val_mae(offline_bundle)
    if val_mae is None or buffer_mae is None:
        return False
    return float(buffer_mae) <= float(META_ONLINE_MAE_MULT) * float(val_mae)


def select_meta_bundle(
    items: list[tuple[Path, dict[str, Any]]],
    *,
    floor: int,
    buffer_mae: float | None,
) -> tuple[Path, dict[str, Any]] | None:
    if not items:
        return None
    offline = [item for item in items if not is_online_bundle(item[0], item[1])]
    online = [item for item in items if is_online_bundle(item[0], item[1])]
    offline.sort(key=lambda item: rank_meta_bundle(item[0], item[1]), reverse=True)
    online.sort(key=lambda item: rank_meta_bundle(item[0], item[1]), reverse=True)
    if not offline:
        return online[0] if online else None
    best_off = offline[0]
    if online:
        best_on = online[0]
        if online_may_replace_offline(best_on[1], best_off[1], buffer_mae=buffer_mae, floor=floor):
            return best_on
    return best_off


def apply_label_scale(raw_edge: float, bundle: dict[str, Any]) -> float:
    raw = bundle.get("label_scale")
    if raw is None:
        return float(raw_edge)
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        return float(raw_edge)
    if abs(scale) <= 1e-12:
        return float(raw_edge)
    return float(raw_edge) * scale


def clamp_meta_edge(edge: float, *, auto_learn: bool = False, n_train: int = 0, floor: int = 0) -> float:
    _ = (auto_learn, n_train, floor)
    return max(META_EDGE_CLIP_LOW, min(META_EDGE_CLIP_HIGH, float(edge)))


def bundle_schema_ok(bundle: dict[str, Any], expected_hash: str) -> bool:
    return bundle_schema_hash_ok(bundle, expected_hash)
