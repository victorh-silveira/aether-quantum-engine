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


def _synthetic_xy(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x_arr = rng.normal(size=(n, FEATURE_DIM))
    y_arr = (x_arr[:, 0] + 0.3 * x_arr[:, 7] > 0.0).astype(np.int32)
    if len(np.unique(y_arr)) < 2:
        y_arr[0] = 0
        y_arr[1] = 1
    return x_arr, y_arr


def main() -> int:
    """Treina LGBMClassifier sintetico e grava .pkl bootstrap."""
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
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=15,
        class_weight="balanced",
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(x_arr, y_arr)
    version = "loss_bootstrap_synth"
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
