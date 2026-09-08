"""Gera artefato bootstrap do loss-classifier em infra/docker/loss-models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import lightgbm as lgb
import numpy as np

from aether_paths import repo_path


FEATURE_DIM = 24


def live_like_synthetic_xy(
    feature_dim: int = FEATURE_DIM,
    *,
    n: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X/y na escala do vetor live 24D (espelha infra/docker/loss-classifier/runtime.py)."""
    rng = np.random.default_rng(int(seed))
    rows = int(n)
    dim = int(feature_dim)
    x_arr = np.zeros((rows, dim), dtype=np.float64)
    x_arr[:, 0] = rng.uniform(0.0, 0.15, size=rows)
    if dim > 1:
        x_arr[:, 1] = rng.uniform(0.42, 0.58, size=rows)
    for idx in (2, 3, 14, 15):
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
        x_arr[:, 13] = rng.uniform(0.35, 0.65, size=rows)
    if dim > 16:
        x_arr[:, 16] = rng.integers(0, 2, size=rows).astype(np.float64)
    if dim > 17:
        x_arr[:, 17] = rng.integers(0, 2, size=rows).astype(np.float64)
    if dim > 18:
        x_arr[:, 18] = rng.uniform(0.42, 0.58, size=rows)
    if dim > 19:
        x_arr[:, 19] = np.clip(rng.normal(0.0, 0.5, size=rows), -3.0, 3.0)
    if dim > 20:
        x_arr[:, 20] = rng.uniform(0.50, 0.65, size=rows)
    if dim > 21:
        x_arr[:, 21] = np.clip(rng.normal(0.0, 0.8, size=rows), -3.0, 3.0)
    if dim > 22:
        x_arr[:, 22] = np.clip(rng.normal(0.0, 0.8, size=rows), -3.0, 3.0)
    if dim > 23:
        x_arr[:, 23] = rng.integers(0, 2, size=rows).astype(np.float64)
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


def _synthetic_xy(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Compat: gera pares live-like para bootstrap offline."""
    return live_like_synthetic_xy(FEATURE_DIM, n=int(n), seed=42)


def main() -> int:
    """Treina LGBMClassifier live-like e grava .pkl bootstrap."""
    parser = argparse.ArgumentParser(description="Bootstrap loss-classifier pkl")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_path("infra", "docker", "loss-models"),
        help="Diretorio de saida dos artefatos",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    x_arr, y_arr = _synthetic_xy()
    model = lgb.LGBMClassifier(
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=4,
        max_depth=3,
        min_child_samples=8,
        class_weight="balanced",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.4,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(x_arr, y_arr)
    version = "loss_bootstrap_live64"
    path = out_dir / f"{version}.pkl"
    joblib.dump(
        {
            "model": model,
            "model_type": "classifier",
            "feature_names": [f"f_{i}" for i in range(FEATURE_DIM)],
            "n_train": int(len(y_arr)),
            "auto_learn_applied": False,
            "bootstrap": True,
            "model_version": version,
            "feature_dim": FEATURE_DIM,
        },
        path,
    )
    print(f"[AETHER] loss-bootstrap ok out=infra/docker/loss-models/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
